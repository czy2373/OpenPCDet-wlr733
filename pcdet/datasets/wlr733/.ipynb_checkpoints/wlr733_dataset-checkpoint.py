import os
import numpy as np
import pickle
from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pcdet.utils import box_utils
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import glob
import tqdm


class WLR733Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path,
                         logger=logger)
        if root_path is None:
            rp = getattr(dataset_cfg, 'DATA_PATH', None) or dataset_cfg.get('DATA_PATH', None)
        else:
            rp = root_path
        assert rp is not None, 'DATA_PATH is not set in dataset config and root_path is None'
        self.root_path = Path(rp)
        split_key = 'train' if training else 'test'
        split_name = dataset_cfg.DATA_SPLIT[split_key]
        split_file = self.root_path / 'ImageSets' / f'{split_name}.txt'
        self.sample_id_list = [x.strip() for x in open(split_file).readlines()] if split_file.exists() else []
        self.infos = []
        info_cfg = getattr(dataset_cfg, 'INFO_PATH', None) or dataset_cfg.get('INFO_PATH', None)
        if info_cfg is not None:
            info_paths = info_cfg.get(split_key, [])
            if isinstance(info_paths, (str, Path)):
                info_paths = [info_paths]

            for ip in info_paths:
                ip = str(ip)
                cand = Path(ip)
                if not cand.exists():
                    cand = Path.cwd() / ip
                if not cand.exists():
                    cand = (self.root_path.parent / ip)
                if not cand.exists():
                    raise FileNotFoundError(f'INFO_PATH not found: {ip}')
                with open(cand, 'rb') as f:
                    self.infos.extend(pickle.load(f))

        if (not self.training) and len(self.infos) == 0:
            raise RuntimeError('No infos loaded for test split; check DATA_CONFIG.INFO_PATH.test in your yaml.')

    def __len__(self):
        return len(self.sample_id_list)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'training' / 'velodyne' / f'{idx}.bin'
        assert lidar_file.exists(), f"{lidar_file} not found!"
        pts = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        return pts

    def get_label(self, idx):
        label_file = self.root_path / 'training' / 'label_2' / f'{idx}.txt'
        if not label_file.exists():
            return np.zeros((0, 8), dtype=np.float32)

        objs = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                name = parts[0].lower()
                if name not in self.class_names:
                    continue
                # KITTI字段顺序: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                ry = float(parts[14])
                objs.append([x, y, z, l, w, h, ry, self.class_names.index(name)])
        if len(objs) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(objs, dtype=np.float32)

    def __getitem__(self, index):
        idx = self.sample_id_list[index]
        points = self.get_lidar(idx)
        annos = self.get_label(idx)
        input_dict = {
            'frame_id': idx,
            'points': points,
        }
        if len(annos) > 0:
            gt_boxes = annos[:, :7].astype(np.float32)
            gt_names = np.array([self.class_names[int(i)] for i in annos[:, 7]], dtype=np.str_)
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = np.array([], dtype=np.str_)
        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def generate_infos(self, split='train'):
        split_dir = self.root_path / 'ImageSets' / f'{split}.txt'
        sample_ids = [x.strip() for x in open(split_dir).readlines()]
        infos = []
        for idx in tqdm.tqdm(sample_ids, desc=f'Generate infos ({split})'):
            info = {
                'point_cloud': {'num_features': 4, 'lidar_idx': idx},
                'annotations': {}
            }
            annos = self.get_label(idx)
            if annos.shape[0] > 0:
                info['annotations']['name'] = [self.class_names[int(i)] for i in annos[:, 7]]
                info['annotations']['gt_boxes_lidar'] = annos[:, :7]
            infos.append(info)
        self.infos = infos
        return infos

    def create_custom_infos(self):
        print(f'Create info files for WLR733 Dataset @ {self.root_path}')
        for split in ['train', 'val']:
            infos = self.generate_infos(split)
            filename = self.root_path / f'wlr733_infos_{split}.pkl'
            with open(filename, 'wb') as f:
                import pickle
                pickle.dump(infos, f)
            print(f'Saved: {filename}, total {len(infos)} samples.')

    def evaluation(self, det_annos, class_names, **kwargs):
        import numpy as np
        """
        KITTI-style evaluation with robust frame-id alignment.
        We normalize both det and gt ids by stripping leading zeros.
        Only evaluate the 'Car' class (vehicle).
        """
        # ---------- 1) Build GT from self.infos ----------
        gt_annos_kitti = []
        gt_ids = []

        for info in self.infos:
            fid = str(info['point_cloud'].get('lidar_idx'))
            gt_ids.append(fid)

            ann = info.get('annotations') or info.get('annos') or {}
            names_raw = ann.get('name', [])
            # 统一成 'Car'
            names = np.array(
                [('Car' if str(n).lower() in ('vehicle', 'car') else str(n)) for n in names_raw],
                dtype=object
            )
            mask = np.array([n == 'Car' for n in names], dtype=bool)
            m = int(mask.sum())

            gt_boxes_lidar = ann.get('gt_boxes_lidar', np.zeros((0, 7), np.float32))
            gt_boxes_lidar = np.asarray(gt_boxes_lidar, dtype=np.float32).reshape(-1, 7)
            if gt_boxes_lidar.shape[0] != len(names):
                # 长度不一致时，按 names 的长度裁剪（你的生成逻辑里通常是一致的）
                n = len(names)
                gt_boxes_lidar = gt_boxes_lidar[:n]

            gt = {
                'name': names[mask] if m > 0 else np.array([], dtype=object),
                # 以下键为兼容 eval.py 的必需键，给零占位（OpenPCDet 3D评估用的是 gt_boxes_lidar）
                'truncated': np.zeros(m, np.float32),
                'occluded': np.zeros(m, np.float32),
                'alpha': np.zeros(m, np.float32),
                'bbox': np.zeros((m, 4), np.float32),
                'dimensions': np.zeros((m, 3), np.float32),
                'location': np.zeros((m, 3), np.float32),
                'rotation_y': np.zeros(m, np.float32),
                'gt_boxes_lidar': gt_boxes_lidar[mask] if m > 0 else np.zeros((0, 7), np.float32),
            }
            gt_annos_kitti.append(gt)

        # ---------- 2) Normalize frame ids ----------
        def _norm(fid):
            s = str(fid)
            s = Path(s).stem  # 防止带扩展名
            s = s.lstrip('0') or '0'
            return s

        # det: 以去零ID为键
        det_by = {}
        for a in det_annos:
            k = _norm(a.get('frame_id', ''))
            # 容错：有的推理没带 frame_id，则从 file_id 等字段兜底
            if not k:
                k = _norm(a.get('file_id', ''))
            det_by[k] = a

        # gt: 以去零ID为键
        gt_by = {_norm(fid): anno for fid, anno in zip(gt_ids, gt_annos_kitti)}

        # ---------- 3) Align on common IDs ----------
        common = sorted(set(det_by.keys()) & set(gt_by.keys()))
        if not common:
            raise RuntimeError(
                "[Eval] No matched frames between det and gt.\n"
                f" det frames (first 5): {sorted(det_by.keys())[:5]}\n"
                f" gt  frames (first 5): {sorted(gt_by.keys())[:5]}\n"
                " Hints: Ensure __getitem__ sets frame_id and val.pkl ids match predictions."
            )
        det_aligned = [det_by[k] for k in common]
        gt_aligned = [gt_by[k] for k in common]

        # ---------- 4) Canonicalize prediction fields ----------
        for d in det_aligned:
            # 类别名统一成 'Car'
            if 'name' in d and d['name'] is not None:
                d['name'] = np.array(
                    ['Car' if str(n).lower() in ('vehicle', 'car') else str(n)
                     for n in d['name']],
                    dtype=object
                )
            else:
                d['name'] = np.array([], dtype=object)

            # 3D boxes字段：boxes_lidar / box3d_lidar / pred_boxes 统一到 boxes_lidar
            if 'boxes_lidar' in d:
                boxes = d['boxes_lidar']
            elif 'box3d_lidar' in d:
                boxes = d['box3d_lidar']
            elif 'pred_boxes' in d:
                boxes = d['pred_boxes']
            else:
                boxes = np.zeros((0, 7), dtype=np.float32)
            d['boxes_lidar'] = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)

            # 分数：scores / pred_scores / score 统一到 score
            scores = None
            for key in ('score', 'scores', 'pred_scores'):
                if key in d:
                    scores = d[key]
                    break
            if scores is None:
                scores = np.zeros((len(d['boxes_lidar']),), dtype=np.float32)
            else:
                scores = np.asarray(scores, dtype=np.float32).reshape(-1, )
                # 和 boxes 数量对齐
                if scores.shape[0] != d['boxes_lidar'].shape[0]:
                    n = min(scores.shape[0], d['boxes_lidar'].shape[0])
                    scores = scores[:n]
                    d['boxes_lidar'] = d['boxes_lidar'][:n]
                    d['name'] = d['name'][:n] if len(d['name']) else np.array([], dtype=object)
            d['score'] = scores

        # ---------- 5) Pad KITTI-required keys (length-consistent) ----------
        def _pad_kitti_keys_gt(a):
            n = len(a.get('name', []))
            a.setdefault('truncated', np.zeros(n, np.float32))
            a.setdefault('occluded', np.zeros(n, np.float32))
            a.setdefault('alpha', np.zeros(n, np.float32))
            a.setdefault('bbox', np.zeros((n, 4), np.float32))
            a.setdefault('dimensions', np.zeros((n, 3), np.float32))
            a.setdefault('location', np.zeros((n, 3), np.float32))
            a.setdefault('rotation_y', np.zeros(n, np.float32))
            # 强制长度对齐
            for k in ('truncated', 'occluded', 'alpha', 'rotation_y', 'bbox', 'dimensions', 'location'):
                v = np.asarray(a[k])
                if v.ndim == 1 and v.shape[0] != n: a[k] = v[:n]
                if v.ndim == 2 and v.shape[0] != n: a[k] = v[:n]

        def _pad_kitti_keys_dt(a):
            n = len(a.get('name', []))
            a.setdefault('alpha', np.zeros(n, np.float32))
            a.setdefault('bbox', np.zeros((n, 4), np.float32))
            a.setdefault('dimensions', np.zeros((n, 3), np.float32))
            a.setdefault('location', np.zeros((n, 3), np.float32))
            a.setdefault('rotation_y', np.zeros(n, np.float32))
            a.setdefault('score', np.asarray(a.get('score', np.zeros(n, np.float32)), np.float32))
            # 强制长度对齐
            for k in ('alpha', 'rotation_y', 'bbox', 'dimensions', 'location', 'score'):
                v = np.asarray(a[k])
                if v.ndim == 1 and v.shape[0] != n: a[k] = v[:n]
                if v.ndim == 2 and v.shape[0] != n: a[k] = v[:n]

        for g in gt_aligned:  _pad_kitti_keys_gt(g)
        for d in det_aligned: _pad_kitti_keys_dt(d)

        z_shift = 1.5  # 可以先试 1.5，再试 -1.5，看哪个让 IoU > 0.5
        for g in gt_aligned:
            if 'gt_boxes_lidar' in g:
                g['gt_boxes_lidar'][:, 2] -= z_shift
        for d in det_aligned:
            if 'boxes_lidar' in d:
                d['boxes_lidar'][:, 2] -= z_shift
        # ---------- 6) Run official KITTI evaluator (Car only) ----------
        print("\n===== [DEBUG] IoU Sanity Check =====")
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        import torch

        if len(common) == 0:
            print("[DEBUG] No common frames between det and gt.")
        else:
            k = common[0]
            d = det_by[k]
            g = gt_by[k]
            if 'boxes_lidar' in d and 'gt_boxes_lidar' in g:
                dt = torch.from_numpy(d['boxes_lidar']).cuda()
                gt = torch.from_numpy(g['gt_boxes_lidar']).cuda()
                ious = iou3d_nms_utils.boxes_iou3d_gpu(dt, gt).cpu().numpy()
                print(
                    f"[DEBUG IoU] Frame={k}, IoU_max={ious.max():.3f}, IoU_mean={ious.mean():.3f}, N_pred={len(dt)}, N_gt={len(gt)}")
            else:
                print(f"[DEBUG] Frame={k} missing boxes_lidar or gt_boxes_lidar")

        def _lidar_to_cam_fields(boxes_lidar: np.ndarray):
            """
            仅用于评估一致性：将 LiDAR 7DoF boxes 映射为“类似相机系”的字段，
            使 det 与 gt 都用同一近似变换，从而 KITTI 评估器能得到非零 IoU。
            注意：这不是 KITTI 真正的标定，仅为自定义数据的一致性评估。
            """
            if boxes_lidar.size == 0:
                return (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
            x = boxes_lidar[:, 0]
            y = boxes_lidar[:, 1]
            z = boxes_lidar[:, 2]
            l = boxes_lidar[:, 3]
            w = boxes_lidar[:, 4]
            h = boxes_lidar[:, 5]
            ry = boxes_lidar[:, 6]

            # 近似到“相机系”：x_cam≈x_fwd, y_cam≈-y_left, z_cam≈z_up(保持)；yaw 取 -ry
            dimensions = np.stack([h, w, l], axis=1).astype(np.float32)  # KITTI: [h, w, l]
            location = np.stack([x, -y, z], axis=1).astype(np.float32)  # 近似到相机系
            rot_y = (-ry).astype(np.float32)  # 近似到相机 yaw

            return dimensions, location, rot_y

        def _fill_from_lidar_boxes_for_dt(a: dict):
            boxes = None
            for key in ['boxes_lidar', 'dt_boxes_lidar', 'boxes_3d', 'boxes']:
                val = a.get(key, None)
                if isinstance(val, (list, np.ndarray)):
                    boxes = np.asarray(val, dtype=np.float32)
                    break

            if boxes is None:
                a.setdefault('dimensions', np.zeros((0, 3), np.float32))
                a.setdefault('location', np.zeros((0, 3), np.float32))
                a.setdefault('rotation_y', np.zeros((0,), np.float32))
                return

            boxes = np.asarray(boxes, dtype=np.float32)
            dims, loc, rot = _lidar_to_cam_fields(boxes)
            a['dimensions'] = dims
            a['location'] = loc
            a['rotation_y'] = rot

            # 名称统一成 Car；分数补齐
            names = a.get('name', [])
            a['name'] = np.array(
                ['Car' if str(n).lower() in ('vehicle', 'car') else str(n) for n in names],
                dtype=object
            )
            if 'score' not in a and 'scores' in a:
                a['score'] = np.asarray(a['scores'], dtype=np.float32)
            if 'score' not in a:
                a['score'] = np.ones((len(a['name']),), dtype=np.float32)

        def _fill_from_lidar_boxes_for_gt(a: dict):
            # 依次尝试这些键，拿到激光坐标系的 GT boxes
            boxes = None
            for key in ['gt_boxes_lidar', 'boxes_lidar', 'gt_boxes', 'boxes']:
                val = a.get(key, None)
                if isinstance(val, (list, np.ndarray)):
                    boxes = np.asarray(val, dtype=np.float32)
                    break

            if boxes is None:
                # 没有 box 时也要把关键字段补齐为空，避免后续 KeyError
                a.setdefault('name', np.array([], dtype=object))
                a.setdefault('truncated', np.zeros((0,), np.float32))
                a.setdefault('occluded', np.zeros((0,), np.float32))
                a.setdefault('alpha', np.zeros((0,), np.float32))
                a.setdefault('bbox', np.zeros((0, 4), np.float32))
                a.setdefault('dimensions', np.zeros((0, 3), np.float32))
                a.setdefault('location', np.zeros((0, 3), np.float32))
                a.setdefault('rotation_y', np.zeros((0,), np.float32))
                return

            # 从激光系 box 计算 KITTI 需要的字段（dims, loc, rot_y）
            boxes = np.asarray(boxes, dtype=np.float32)
            dims, loc, rot = _lidar_to_cam_fields(boxes)
            a['dimensions'] = dims
            a['location'] = loc
            a['rotation_y'] = rot

            # 名称统一成 Car
            names = a.get('name', [])
            a['name'] = np.array(
                ['Car' if str(n).lower() in ('vehicle', 'car') else str(n) for n in names],
                dtype=object
            )

            # 其余 KITTI 必需字段占位（避免 eval 中 KeyError）
            n = len(a['name'])
            a.setdefault('truncated', np.zeros((n,), np.float32))
            a.setdefault('occluded', np.zeros((n,), np.float32))
            a.setdefault('alpha', np.zeros((n,), np.float32))
            a.setdefault('bbox', np.zeros((n, 4), np.float32))

        # —— 对齐好的 det/gt 逐个填充相机字段（用 LiDAR 近似）——
        for d in det_aligned:
            _fill_from_lidar_boxes_for_dt(d)

        for g in gt_aligned:
            _fill_from_lidar_boxes_for_gt(g)

               # ---------- 7) Evaluate in LiDAR coordinates directly ----------
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        import torch

        all_iou = []
        tp, fp, fn = 0, 0, 0

        for d, g in zip(det_aligned, gt_aligned):
            boxes_pred = torch.from_numpy(d['boxes_lidar']).cuda()
            boxes_gt = torch.from_numpy(g['gt_boxes_lidar']).cuda()
            if boxes_pred.shape[0] == 0 or boxes_gt.shape[0] == 0:
                continue
            iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes_pred, boxes_gt).cpu().numpy()
            all_iou.append(iou)

            max_iou = iou.max(axis=1) if iou.size > 0 else np.zeros(0)
            tp += (max_iou > 0.5).sum()
            fp += (max_iou <= 0.5).sum()
            fn += max(0, boxes_gt.shape[0] - (max_iou > 0.5).sum())

        mean_iou = np.mean([i.max() for i in all_iou if i.size > 0]) if all_iou else 0
        recall = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)

        result_str = f"\n[WLR733 Eval in LiDAR Coord]\n" \
                     f"Mean IoU: {mean_iou:.3f}\n" \
                     f"TP={tp}, FP={fp}, FN={fn}\n" \
                     f"Recall={recall:.3f}, Precision={precision:.3f}\n"
        result_dict = {'mean_iou': mean_iou, 'recall': recall, 'precision': precision}

        print(result_str)
        return result_str, result_dict



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    cfg_file = 'tools/cfgs/dataset_configs/wlr733_dataset.yaml'
    with open(cfg_file, 'r') as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))
    dataset = WLR733Dataset(
        dataset_cfg=dataset_cfg,
        class_names=dataset_cfg.CLASS_NAMES,
        root_path=Path('data/wlr733'),
        training=True
    )
    dataset.create_custom_infos()
