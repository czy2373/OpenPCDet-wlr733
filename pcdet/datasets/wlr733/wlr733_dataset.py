import os
import numpy as np
import pickle
from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pcdet.utils import box_utils
# from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
import glob
import tqdm
import cv2
import re



class WLR733Dataset(DatasetTemplate):
    # ====== 把你类里的 __init__ 整段替换成下面这份（只保留这一份） ======
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)

        rp = root_path if root_path is not None else (getattr(dataset_cfg, 'DATA_PATH', None) or dataset_cfg.get('DATA_PATH', None))
        assert rp is not None, 'DATA_PATH is not set in dataset config and root_path is None'
        self.root_path = self._resolve_dataset_path(rp)

        split_key = 'train' if training else 'test'
        split_name = dataset_cfg.DATA_SPLIT[split_key]
        split_file = self.root_path / 'ImageSets' / f'{split_name}.txt'
        self.sample_id_list = [x.strip() for x in open(split_file).readlines()] if split_file.exists() else []

        # ===== Image cfg =====
        self.image_root = self._resolve_image_root(getattr(dataset_cfg, 'IMAGE_ROOT', None))
        self.image_resize_hw = tuple(getattr(dataset_cfg, 'IMAGE_RESIZE_HW', [1080, 1920]))  # [H, W]
        self.image_ext = getattr(dataset_cfg, 'IMAGE_EXT', '.jpg')
        self.image_exts = ['.jpg', '.png']  # 统一成 list，避免字符串迭代 bug

        # ===== Calibration =====
        calib_path = self.root_path / 'training' / 'calib' / 'calib.npz'
        if calib_path.exists():
            calib = np.load(str(calib_path))
            self.cam_K = calib['K'].astype(np.float32)                    # (3,3)
            self.T_cam_from_lidar = calib['T_cam_from_lidar'].astype(np.float32)  # (4,4)

            # ===== Runtime refine: dpitch + dz (用环境变量或yaml可控) =====
            refine = getattr(dataset_cfg, 'CALIB_REFINE', None) or dataset_cfg.get('CALIB_REFINE', None) or {}
            dpitch_deg = float(refine.get('DPITCH_DEG', os.environ.get('WLR733_DPITCH_DEG', 0.0)))
            dz_lidar_m = float(refine.get('DZ_LIDAR_M', os.environ.get('WLR733_DZ_LIDAR_M', 0.0)))

            self.T_cam_from_lidar = self._apply_extrinsic_refine(
                self.T_cam_from_lidar, dpitch_deg=dpitch_deg, dz_lidar_m=dz_lidar_m
            )
        else:
            self.cam_K = None
            self.T_cam_from_lidar = None

        # ===== Load infos =====
        self.infos = []
        info_cfg = getattr(dataset_cfg, 'INFO_PATH', None) or dataset_cfg.get('INFO_PATH', None)
        if info_cfg is not None:
            info_paths = info_cfg.get(split_key, [])
            if isinstance(info_paths, (str, Path)):
                info_paths = [info_paths]
            for ip in info_paths:
                cand = Path(str(ip))
                if not cand.exists():
                    cand = Path.cwd() / str(ip)
                if not cand.exists():
                    cand = (self.root_path.parent / str(ip))
                if not cand.exists():
                    raise FileNotFoundError(f'INFO_PATH not found: {ip}')
                with open(cand, 'rb') as f:
                    self.infos.extend(pickle.load(f))

        if (not self.training) and len(self.infos) == 0:
            raise RuntimeError('No infos loaded for test split; check DATA_CONFIG.INFO_PATH.test in your yaml.')

        self.info_by_id = {}
        self._rebuild_info_index()

        # ===== Filter frames without matched images =====
        if self.image_root.exists():
            old_n = len(self.sample_id_list)
            old_ids = list(self.sample_id_list)
            valid = []
            for sid in self.sample_id_list:
                if self._find_image_path(sid) is not None:
                    valid.append(sid)
            self.sample_id_list = valid
            msg = f'[WLR733Dataset] keep {len(valid)}/{old_n} samples with images under {self.image_root}'
            if old_n > 0 and len(valid) == 0:
                msg += f'; first_ids={old_ids[:5]}'
            (logger.info(msg) if logger is not None else print(msg))


    # ====== 用下面这个替换你类里的 _find_image_path ======
    def _find_image_path(self, sid: str):
        _, image_path = self._resolve_info_image_path(sid)
        return image_path

    @staticmethod
    def _norm_key(sid: str):
        sid = str(sid).strip().split('/')[-1].split('\\')[-1]
        sid = sid.split('.')[0]
        return sid

    def _rebuild_info_index(self):
        self.info_by_id = {}
        for info in self.infos:
            sid = info.get('point_cloud', {}).get('lidar_idx', info.get('frame_id', ''))
            sid = self._norm_key(sid)
            if sid:
                self.info_by_id[sid] = info

    @staticmethod
    def _resolve_dataset_path(path_like):
        path = Path(path_like)
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return path

    def _resolve_image_root(self, image_root_cfg):
        if image_root_cfg is None:
            return self.root_path / 'image'

        image_root = Path(image_root_cfg)
        if image_root.is_absolute():
            return image_root

        cwd_candidate = Path.cwd() / image_root
        if cwd_candidate.exists():
            return cwd_candidate

        data_local_candidate = self.root_path / 'image'
        if data_local_candidate.exists():
            return data_local_candidate

        return cwd_candidate

    def _get_info(self, sid: str):
        return self.info_by_id.get(self._norm_key(sid), None)

    def _resolve_cam_frame_id(self, sid: str, info=None, sync_map=None):
        info = self._get_info(sid) if info is None else info
        if info is not None:
            image_meta = info.get('image', {}) or {}
            sync_meta = info.get('sync', {}) or {}
            for candidate in (image_meta.get('cam_frame_id', None), sync_meta.get('cam_frame_id', None)):
                if candidate not in (None, ''):
                    return self._norm_key(candidate)

        key = self._norm_key(sid)
        if sync_map is not None:
            mapped = sync_map.get(key, None)
            if mapped not in (None, ''):
                return self._norm_key(mapped)

        return key

    def _resolve_image_candidate(self, image_path_like):
        if image_path_like in (None, ''):
            return None

        image_path = Path(str(image_path_like))
        if image_path.is_absolute():
            return image_path
        if self.image_root is None:
            return None
        return self.image_root / image_path

    def _normalize_image_path_for_info(self, image_path_like, abs_path: Path):
        image_path = Path(str(image_path_like))
        if not image_path.is_absolute():
            return image_path.as_posix()

        if self.image_root is not None:
            try:
                return abs_path.relative_to(self.image_root).as_posix()
            except ValueError:
                pass

        return abs_path.as_posix()

    def _resolve_info_image_path(self, sid: str, info=None, cam_frame_id=None):
        info = self._get_info(sid) if info is None else info
        cam_frame_id = self._resolve_cam_frame_id(sid, info=info) if cam_frame_id is None else self._norm_key(cam_frame_id)

        image_meta = info.get('image', {}) if info is not None else {}
        image_path_raw = image_meta.get('image_path', None)
        if image_path_raw not in (None, ''):
            abs_path = self._resolve_image_candidate(image_path_raw)
            if abs_path is not None and abs_path.exists():
                return self._normalize_image_path_for_info(image_path_raw, abs_path), abs_path

        if self.image_root is not None and cam_frame_id not in (None, ''):
            for ext in self.image_exts:
                rel_path = f'{cam_frame_id}{ext}'
                abs_path = self.image_root / rel_path
                if abs_path.exists():
                    return rel_path, abs_path

        if image_path_raw not in (None, ''):
            return Path(str(image_path_raw)).as_posix(), self._resolve_image_candidate(image_path_raw)

        return None, None

    @staticmethod
    def _load_image_shape(img_path: Path):
        if img_path is None or (not img_path.exists()):
            return None

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w = img.shape[:2]
        return [int(h), int(w)]

    def _build_sync_meta(self, lidar_frame_id: str, cam_frame_id: str, sync_source=None):
        if sync_source not in (None, ''):
            source = str(sync_source)
        elif cam_frame_id == lidar_frame_id:
            source = 'identity_sync_filename'
        else:
            source = 'external_sync_map'

        return {
            'lidar_frame_id': str(lidar_frame_id),
            'cam_frame_id': str(cam_frame_id),
            'source': source,
        }

    def _build_calib_meta(self, sid: str):
        key = self._norm_key(sid)
        calib_meta = {
            'candidate_name': 'native',
        }

        calib_file = self.root_path / 'training' / 'calib' / f'{key}.npz'
        if calib_file.exists():
            calib = np.load(str(calib_file))
            calib_meta.update({
                'cam_K': calib['K'].astype(np.float32),
                'T_cam_from_lidar': calib['T_cam_from_lidar'].astype(np.float32),
                'calib_source': 'per_frame_npz',
                'cam_k_source': 'per_frame_npz',
            })
            return calib_meta

        if self.cam_K is not None:
            calib_meta['cam_K'] = self.cam_K.astype(np.float32).copy()
        if self.T_cam_from_lidar is not None:
            calib_meta['T_cam_from_lidar'] = self.T_cam_from_lidar.astype(np.float32).copy()
        if ('cam_K' in calib_meta) or ('T_cam_from_lidar' in calib_meta):
            calib_meta.update({
                'calib_source': 'global_calib_npz',
                'cam_k_source': 'global_calib_npz',
            })
        else:
            calib_meta.update({
                'calib_source': 'missing',
                'cam_k_source': 'missing',
            })

        return calib_meta

    def _load_frame_calib(self, sid: str):
        info = self._get_info(sid)
        if info is not None:
            calib = info.get('calib', None)
            if calib is not None:
                cam_k = calib.get('cam_K', None)
                t_cam_from_lidar = calib.get('T_cam_from_lidar', None)
                if cam_k is not None and t_cam_from_lidar is not None:
                    return np.asarray(cam_k, dtype=np.float32), np.asarray(t_cam_from_lidar, dtype=np.float32)

        key = self._norm_key(sid)
        calib_file = self.root_path / 'training' / 'calib' / f'{key}.npz'
        if calib_file.exists():
            calib = np.load(str(calib_file))
            return calib['K'].astype(np.float32), calib['T_cam_from_lidar'].astype(np.float32)

        return self.cam_K, self.T_cam_from_lidar


    # ====== 只保留一份 _read_image_chw_float（你现在重复定义了两次） ======
    def _read_image_chw_float(self, img_path: Path, resize_hw):
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        H, W = resize_hw  # [H,W]
        if (H is not None) and (W is not None):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img, (H if H else H0), (W if W else W0)


    # ====== 用下面这个替换你类里的 __getitem__ ======
    def __getitem__(self, index):
        idx = self.sample_id_list[index]
        points = self.get_lidar(idx)
        annos = self.get_label(idx)
        info = self._get_info(idx)

        input_dict = {
            'frame_id': idx,
            'points': points,
        }

        H_fix, W_fix = self.image_resize_hw  # (1080,1920)
        img_path = self._find_image_path(idx)
        cam_K_frame, T_cam_from_lidar_frame = self._load_frame_calib(idx)

        if img_path is not None:
            img, H_img, W_img = self._read_image_chw_float(img_path, (H_fix, W_fix))
            if img is None:
                img = np.zeros((3, H_fix, W_fix), dtype=np.float32)
                has = 0
            else:
                has = 1

            input_dict['images'] = img
            input_dict['image_H'] = H_fix
            input_dict['image_W'] = W_fix
            input_dict['has_image'] = has

            if cam_K_frame is not None:
                input_dict['cam_K'] = cam_K_frame.astype(np.float32)
            if T_cam_from_lidar_frame is not None:
                input_dict['T_cam_from_lidar'] = T_cam_from_lidar_frame.astype(np.float32)
        else:
            # 理论上你已过滤掉无图帧，这里只是兜底
            input_dict['has_image'] = 0
            input_dict['images'] = np.zeros((3, H_fix, W_fix), dtype=np.float32)
            input_dict['image_H'] = H_fix
            input_dict['image_W'] = W_fix
            if cam_K_frame is not None:
                input_dict['cam_K'] = cam_K_frame.astype(np.float32)
            if T_cam_from_lidar_frame is not None:
                input_dict['T_cam_from_lidar'] = T_cam_from_lidar_frame.astype(np.float32)

        if info is not None:
            image_meta = info.get('image', {}) or {}
            sync_meta = info.get('sync', {}) or {}
            calib_meta = info.get('calib', {}) or {}
            input_dict['image_paths'] = np.array(image_meta.get('image_path', '') or '', dtype=np.str_)
            input_dict['cam_frame_id'] = np.array(
                image_meta.get('cam_frame_id', sync_meta.get('cam_frame_id', '')) or '',
                dtype=np.str_,
            )
            input_dict['candidate_name'] = np.array(calib_meta.get('candidate_name', '') or '', dtype=np.str_)
            input_dict['calib_source'] = np.array(calib_meta.get('calib_source', '') or '', dtype=np.str_)

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
    

    @staticmethod
    def _apply_extrinsic_refine(T_cam_from_lidar: np.ndarray,
                                dpitch_deg: float = 0.0,
                                dz_lidar_m: float = 0.0) -> np.ndarray:
        """
        T_cam_from_lidar: 4x4, 将 lidar 点变换到 camera 坐标系
        dpitch_deg: 以 camera 的 x 轴做微小俯仰修正（左乘）
        dz_lidar_m: 在 lidar 的 z 轴方向做高度微调（等价于 t += R*[0,0,dz]）
        """
        if T_cam_from_lidar is None:
            return None

        T = T_cam_from_lidar.astype(np.float32).copy()
        R = T[:3, :3]

        # 1) dz：沿 lidar-z(上) 平移
        if abs(dz_lidar_m) > 1e-8:
            T[:3, 3] = T[:3, 3] + (R @ np.array([0.0, 0.0, dz_lidar_m], dtype=np.float32))

        # 2) dpitch：绕 camera-x 旋转（左乘，修正相机俯仰）
        if abs(dpitch_deg) > 1e-8:
            a = np.deg2rad(dpitch_deg)
            ca, sa = float(np.cos(a)), float(np.sin(a))
            Rx = np.array([[1.0, 0.0, 0.0],
                           [0.0, ca, -sa],
                           [0.0, sa,  ca]], dtype=np.float32)
            T_delta = np.eye(4, dtype=np.float32)
            T_delta[:3, :3] = Rx
            T = T_delta @ T

        return T
    
    def __len__(self):
        return len(self.sample_id_list)

    def get_lidar(self, idx):
        """
        Robust lidar loader for wlr733:
        - idx may be '004620' or '1802/004620'
        - velodyne may contain subfolders: training/velodyne/1802/004620.bin
        """
        import glob
        idx_raw = str(idx).strip().replace('\\', '/')
        key = idx_raw.split('/')[-1].split('.')[0]  # '004620'

        base = self.root_path / 'training' / 'velodyne'

        # 1) direct flat: velodyne/004620.bin
        cand = base / f'{key}.bin'
        if cand.exists():
            return np.fromfile(str(cand), dtype=np.float32).reshape(-1, 4)

        # 2) if idx includes folder: velodyne/1802/004620.bin
        if '/' in idx_raw:
            cand2 = base / idx_raw
            if cand2.suffix != '.bin':
                cand2 = cand2.with_suffix('.bin')
            if cand2.exists():
                return np.fromfile(str(cand2), dtype=np.float32).reshape(-1, 4)

        # 3) search one-level subfolders: velodyne/*/004620.bin
        hits = glob.glob(str(base / '*' / f'{key}.bin'))
        if len(hits) == 1:
            return np.fromfile(hits[0], dtype=np.float32).reshape(-1, 4)
        if len(hits) > 1:
            raise FileNotFoundError(f"Multiple lidar candidates for {key}: {hits[:5]} ...")

        raise FileNotFoundError(f"Lidar bin not found for idx={idx_raw}. Tried: {cand} and {base}/*/{key}.bin")


    def get_label(self, idx):
        idx_raw = str(idx).strip().replace('\\', '/')
        key = idx_raw.split('/')[-1].split('.')[0]  # 000000

        label_dir = self.root_path / 'training' / 'label_2'
        cand1 = label_dir / f'{idx_raw}.txt'   # 兼容：label_2/1802/000000.txt
        cand2 = label_dir / f'{key}.txt'       # 你的实际格式：label_2/000000.txt

        label_file = cand1 if cand1.exists() else cand2
        if not label_file.exists():
            return np.zeros((0, 8), dtype=np.float32)

        objs = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                name = parts[0].lower()
                # 容错：有些标注写 car/Car，这里统一映射到你的 vehicle 类
                if name == 'car' and 'vehicle' in self.class_names:
                    name = 'vehicle'

                if name not in self.class_names:
                    continue

                # KITTI: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                ry = float(parts[14])

                objs.append([x, y, z, l, w, h, ry, self.class_names.index(name)])

        if len(objs) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(objs, dtype=np.float32)

    


    def generate_infos(self, split='train', sync_map=None, sync_source=None):
        split_dir = self.root_path / 'ImageSets' / f'{split}.txt'
        sample_ids = [x.strip() for x in open(split_dir).readlines()]
        sync_lookup = None
        if sync_map is not None:
            sync_lookup = {self._norm_key(k): v for k, v in sync_map.items()}

        infos = []
        for idx in tqdm.tqdm(sample_ids, desc=f'Generate infos ({split})'):
            lidar_frame_id = self._norm_key(idx)
            cam_frame_id = self._resolve_cam_frame_id(idx, sync_map=sync_lookup)
            image_rel_path, image_abs_path = self._resolve_info_image_path(
                idx, info=None, cam_frame_id=cam_frame_id
            )
            info = {
                'frame_id': idx,
                'point_cloud': {'num_features': 4, 'lidar_idx': idx},
                'image': {
                    'cam_frame_id': cam_frame_id,
                    'image_path': image_rel_path,
                    'image_shape': self._load_image_shape(image_abs_path),
                },
                'sync': self._build_sync_meta(
                    lidar_frame_id=lidar_frame_id,
                    cam_frame_id=cam_frame_id,
                    sync_source=sync_source,
                ),
                'calib': self._build_calib_meta(idx),
                'annotations': {}
            }
            annos = self.get_label(idx)
            if annos.shape[0] > 0:
                info['annotations']['name'] = [self.class_names[int(i)] for i in annos[:, 7]]
                info['annotations']['gt_boxes_lidar'] = annos[:, :7]
            infos.append(info)
        self.infos = infos
        self._rebuild_info_index()
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
        # 从 cfg 读取开关；没有就默认 True（你原来就是在用官方评测）
        use_kitti = getattr(self.dataset_cfg, 'USE_KITTI_OFFICIAL', True)

        official_txt = None
        if use_kitti:
            try:
                # 延迟导入，只有真的要跑官方评测时才 import
                from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
            except Exception as e:
                print(f"[WLR733Dataset] Skip KITTI official eval due to error: {e}")
                use_kitti = False
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

        # z_shift: 仅修正 GT 的 z 坐标，用于补偿标注坐标系与点云坐标系之间的高度偏差。
        # 如果你的 label 已经是 LiDAR 系，可以把 z_shift 设为 0.0。
        # 只移动 GT，不移动 Pred；两者同时移动没有意义（对 IoU 无影响）。
        z_shift = 0.0  # 先设 0.0，确认 IoU 正常后如有偏差再调整（正值=GT往下移）
        if abs(z_shift) > 1e-6:
            for g in gt_aligned:
                if 'gt_boxes_lidar' in g and g['gt_boxes_lidar'].shape[0] > 0:
                    g['gt_boxes_lidar'][:, 2] -= z_shift
        
        # ---------- 6) Debug IoU sanity check (robust, never crash) ----------
        print("\n===== [DEBUG] IoU Sanity Check =====")
        try:
            import torch
            from pcdet.ops.iou3d_nms import iou3d_nms_utils
            _iou3d_ok = True
            _iou3d_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            _iou3d_ok = False
            print(f"[DEBUG IoU] skip (iou3d_nms_utils not available): {e}")

        if (not _iou3d_ok) or len(common) == 0:
            print("[DEBUG IoU] skip (iou3d unavailable / no common)")
        else:
            # 找到第一帧 pred/gt 都非空的，避免空数组 max/mean
            picked = None
            for k in common:
                d = det_by.get(k, {})
                g = gt_by.get(k, {})
                if ('boxes_lidar' not in d) or ('gt_boxes_lidar' not in g):
                    continue
                if len(d.get('boxes_lidar', [])) == 0 or len(g.get('gt_boxes_lidar', [])) == 0:
                    continue
                picked = k
                break

            if picked is None:
                k = common[0]
                n_pred = len(det_by.get(k, {}).get('boxes_lidar', []))
                n_gt = len(gt_by.get(k, {}).get('gt_boxes_lidar', []))
                print(f"[DEBUG IoU] no non-empty matched frame found. example Frame={k}, N_pred={n_pred}, N_gt={n_gt}")
            else:
                k = picked
                try:
                    import numpy as np
                    dt = torch.from_numpy(det_by[k]['boxes_lidar']).to(_iou3d_device)
                    gt = torch.from_numpy(gt_by[k]['gt_boxes_lidar']).to(_iou3d_device)
                    ious = iou3d_nms_utils.boxes_iou3d_gpu(dt, gt).detach().cpu().numpy()
                    if ious.size == 0:
                        print(f"[DEBUG IoU] Frame={k}, IoU empty. N_pred={dt.shape[0]}, N_gt={gt.shape[0]}")
                    else:
                        print(f"[DEBUG IoU] Frame={k}, IoU_max={float(np.max(ious)):.3f}, IoU_mean={float(np.mean(ious)):.3f}, N_pred={dt.shape[0]}, N_gt={gt.shape[0]}")
                except Exception as e:
                    print(f"[DEBUG IoU] sanity failed at Frame={k}: {e}")

        # ---------- 7) Evaluate in LiDAR coordinates directly (optional) ----------
        # 这段不是必须项：没有 iou3d 算子就跳过，保证 test.py 能正常结束
        try:
            import torch
            from pcdet.ops.iou3d_nms import iou3d_nms_utils
            _eval_ok = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            _eval_ok = False
            print(f"[WLR733Dataset] Skip LiDAR IoU eval (iou3d_nms_utils not available): {e}")

        if not _eval_ok:
            result_str = "\n[WLR733 Eval in LiDAR Coord]\nSkip (iou3d not available)\n"
            result_dict = {'mean_iou': 0.0, 'recall': 0.0, 'precision': 0.0}
            print(result_str)
            return result_str, result_dict

        import numpy as np
        all_iou = []
        tp, fp, fn = 0, 0, 0

        for d, g in zip(det_aligned, gt_aligned):
            boxes_pred = torch.from_numpy(d.get('boxes_lidar', np.zeros((0, 7), np.float32))).to(device)
            boxes_gt = torch.from_numpy(g.get('gt_boxes_lidar', np.zeros((0, 7), np.float32))).to(device)
            if boxes_pred.shape[0] == 0 or boxes_gt.shape[0] == 0:
                continue
            try:
                iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes_pred, boxes_gt).detach().cpu().numpy()
            except Exception as e:
                print(f"[WLR733Dataset] LiDAR IoU eval failed: {e}")
                break

            if iou.size == 0:
                continue

            all_iou.append(iou)
            max_iou = iou.max(axis=1)
            tp_i = int((max_iou > 0.5).sum())
            tp += tp_i
            fp += int((max_iou <= 0.5).sum())
            fn += max(0, int(boxes_gt.shape[0] - tp_i))

        mean_iou = float(np.mean([i.max() for i in all_iou if i.size > 0])) if all_iou else 0.0
        recall = float(tp / (tp + fn + 1e-6))
        precision = float(tp / (tp + fp + 1e-6))

        result_str = (
            f"\n[WLR733 Eval in LiDAR Coord]\n"
            f"Mean IoU: {mean_iou:.3f}\n"
            f"TP={tp}, FP={fp}, FN={fn}\n"
            f"Recall={recall:.3f}, Precision={precision:.3f}\n"
        )
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
