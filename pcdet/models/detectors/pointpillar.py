from .detector3d_template import Detector3DTemplate
import os
import cv2
import numpy as np
import torch
from pcdet.models.backbones_2d.image_to_bev import ImageToBEV
from pcdet.models.backbones_2d.bev_fusion import BEVFusion

import torch.nn as nn
import torch.nn.functional as F


class TinyImgBackbone(nn.Module):
    def __init__(self, out_channels=64, downsample=4):
        super().__init__()
        assert downsample in [4, 8]
        c1, c2, c3 = 32, 64, out_channels
        layers = []
        layers += [nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)]
        if downsample == 8:
            layers += [nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True)]
        else:
            layers += [nn.Conv2d(c2, c3, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels
        self.downsample = downsample

    def forward(self, x):
        return self.net(x)


class CalibRefinerXY(nn.Module):
    def __init__(self, c_lidar=64, c_img=64, max_shift_m=1.0):
        super().__init__()
        self.max_shift_m = float(max_shift_m)

        self.reduce_l = nn.Sequential(
            nn.Conv2d(c_lidar, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.reduce_i = nn.Sequential(
            nn.Conv2d(c_img, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, lidar_bev, img_bev):
        x = torch.cat([self.reduce_l(lidar_bev), self.reduce_i(img_bev)], dim=1)
        x = self.fuse(x)
        d = self.head(x)                       # (B,2)
        d = self.max_shift_m * torch.tanh(d)   # clamp to [-max_shift_m, +max_shift_m]
        return d


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.dataset_cfg = getattr(dataset, 'dataset_cfg', None)

        # ===== A2.5: 在 map_to_bev 之后（spatial_features）做一次可学习门控融合 =====
        self.use_fusion = bool(getattr(self.model_cfg, "USE_FUSION", False))

        if self.use_fusion:
            # 1) 图像 backbone：输出 img_feat
            img_cfg = getattr(self.model_cfg, "IMAGE_BACKBONE", None) or {}
            img_c = int(getattr(img_cfg, "OUT_CHANNELS", img_cfg.get("OUT_CHANNELS", 64))) if isinstance(img_cfg, dict) else 64
            img_down = int(getattr(img_cfg, "DOWNSAMPLE", img_cfg.get("DOWNSAMPLE", 4))) if isinstance(img_cfg, dict) else 4
            self.img_backbone = TinyImgBackbone(out_channels=img_c, downsample=img_down)

            # 2) Image -> BEV：严格按 POINT_CLOUD_RANGE/VOXEL_SIZE + lidar_bev(H,W) 定义 BEV 网格
            itb_cfg = getattr(self.model_cfg, "IMAGE_TO_BEV", None) or {}
            itb_cfg = dict(itb_cfg) if isinstance(itb_cfg, dict) else itb_cfg

            # 以 yaml 为主：IMAGE_TO_BEV 里没写的字段，自动从 dataset_cfg 补齐，保证与 voxelization 完全一致
            dataset_cfg = getattr(dataset, 'dataset_cfg', None)

            def _ds_get_pcr(dcfg):
                default = [-112.0, -106.0, -12.0, 99.2, 158.0, 2.0]
                if dcfg is None:
                    return default
                if isinstance(dcfg, dict):
                    return dcfg.get('POINT_CLOUD_RANGE', default)
                return getattr(dcfg, 'POINT_CLOUD_RANGE', default)

            def _ds_get_voxel_xy(dcfg):
                default = [0.2, 0.2, 10]
                try:
                    proc = dcfg.get('DATA_PROCESSOR', []) if isinstance(dcfg, dict) else getattr(dcfg, 'DATA_PROCESSOR', [])
                    for p in proc:
                        name = p.get('NAME') if isinstance(p, dict) else getattr(p, 'NAME', None)
                        if name == 'transform_points_to_voxels':
                            vs = p.get('VOXEL_SIZE') if isinstance(p, dict) else getattr(p, 'VOXEL_SIZE', None)
                            if vs is not None:
                                return vs
                except Exception:
                    pass
                return default

            if isinstance(itb_cfg, dict):
                itb_cfg.setdefault('VOXEL_SIZE', _ds_get_voxel_xy(dataset_cfg))
                itb_cfg.setdefault('POINT_CLOUD_RANGE', _ds_get_pcr(dataset_cfg))
                itb_cfg.setdefault('Z0', -2.0)
                itb_cfg.setdefault('Z_LIST', [])
                itb_cfg.setdefault('IMG_DOWNSAMPLE', 4)
                itb_cfg.setdefault('BEV_DOWNSAMPLE', 4)
                itb_cfg.setdefault('DEBUG_SAVE', False)


            self.img_to_bev = ImageToBEV(model_cfg=itb_cfg)

            # 兼容两种名字
            self.img_to_bev.pc_range = itb_cfg['POINT_CLOUD_RANGE']
            self.img_to_bev.point_cloud_range = itb_cfg['POINT_CLOUD_RANGE']
            self.point_cloud_range = itb_cfg['POINT_CLOUD_RANGE']

            # 3) BEV 融合
            fusion_cfg = getattr(self.model_cfg, "BEV_FUSION", None) or {"C_IMG": img_c, "ALPHA_INIT": 0.005}
            from easydict import EasyDict

            fusion_cfg = getattr(self.model_cfg, "BEV_FUSION", None) or {}
            if isinstance(fusion_cfg, dict):
                fusion_cfg = EasyDict(fusion_cfg)

            # 给个兜底：IMG_CHANNELS 没写就用 img_c
            if not hasattr(fusion_cfg, "IMG_CHANNELS"):
                fusion_cfg.IMG_CHANNELS = img_c

            self.bev_fusion = BEVFusion(fusion_cfg)


            # ===== Calib refine (XY only) + ROI structure alignment loss =====
            cr_cfg = getattr(self.model_cfg, "CALIB_REFINE", None) or {}

            def _cfg_get(cfg, k, d):
                if cfg is None:
                    return d
                if isinstance(cfg, dict):
                    return cfg.get(k, d)
                return getattr(cfg, k, d)

            self.use_calib_refine = bool(_cfg_get(cr_cfg, "ENABLE", False))
            self.calib_max_shift_m = float(_cfg_get(cr_cfg, "MAX_SHIFT_M", 1.0))
            self.loss_align_w = float(_cfg_get(cr_cfg, "LOSS_ALIGN_W", 0.2))
            self.loss_reg_w = float(_cfg_get(cr_cfg, "LOSS_REG_W", 0.01))
            self.use_valid_mask = bool(_cfg_get(cr_cfg, "USE_VALID_MASK", True))
            self.detach_lidar_for_align = bool(_cfg_get(cr_cfg, "DETACH_LIDAR", True))
            self.detach_img0_for_refiner = bool(_cfg_get(cr_cfg, "DETACH_IMG0", True))
            self.detach_img_feat_for_proj = bool(_cfg_get(cr_cfg, "DETACH_IMG_FEAT", True))

            if self.use_calib_refine:
                self.calib_refiner = CalibRefinerXY(
                    c_lidar=64, c_img=img_c, max_shift_m=self.calib_max_shift_m
                )
            else:
                self.calib_refiner = None

            # ===== ROI config: fusion ROI + track ROI（全部配置化，禁止硬编码） =====
            roi_cfg = getattr(self.model_cfg, "ROI", None) or {}
            self.fusion_roi_cfg = self._norm_roi_cfg(
                self._cfg_get(roi_cfg, "FUSION", None),
                default_mode="corridor",
                default_corridor={
                    "K": 1.53,
                    "B": 18.5,
                    "X0": -80.0,
                    "X1": 90.0,
                    "HALF_LEFT": 12.0,
                    "HALF_RIGHT": 6.0,
                },
                default_enabled=True,
            )
            self.track_roi_cfg = self._norm_roi_cfg(
                self._cfg_get(roi_cfg, "TRACK", None),
                default_mode="corridor",
                default_corridor={
                    "K": 1.53,
                    "B": 18.5,
                    "X0": -80.0,
                    "X1": 90.0,
                    "HALF_LEFT": 9.0,
                    "HALF_RIGHT": 3.5,
                },
                default_enabled=True,
            )

            # sobel kernels (buffers)
            kx = torch.tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            ky = torch.tensor([[-1., -2., -1.],
                               [0.,  0.,  0.],
                               [1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("_sobel_kx", kx)
            self.register_buffer("_sobel_ky", ky)

            self._loss_calib = None
            self._tb_calib = {}

            self.apply_freeze()

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _norm_roi_cfg(self, cfg, default_mode, default_corridor, default_enabled=True):
        cfg = cfg or {}
        enabled = bool(self._cfg_get(cfg, "ENABLED", default_enabled))
        mode = str(self._cfg_get(cfg, "MODE", default_mode)).lower()

        corridor = self._cfg_get(cfg, "CORRIDOR", {}) or {}
        corridor = {
            "K": float(self._cfg_get(corridor, "K", default_corridor["K"])),
            "B": float(self._cfg_get(corridor, "B", default_corridor["B"])),
            "X0": float(self._cfg_get(corridor, "X0", default_corridor["X0"])),
            "X1": float(self._cfg_get(corridor, "X1", default_corridor["X1"])),
            "HALF_LEFT": float(self._cfg_get(corridor, "HALF_LEFT", default_corridor["HALF_LEFT"])),
            "HALF_RIGHT": float(self._cfg_get(corridor, "HALF_RIGHT", default_corridor["HALF_RIGHT"])),
        }

        polygon = self._cfg_get(cfg, "POLYGON", []) or []
        polygon = [[float(p[0]), float(p[1])] for p in polygon if isinstance(p, (list, tuple)) and len(p) >= 2]

        if mode == "polygon" and len(polygon) < 3:
            mode = "corridor"

        return {
            "ENABLED": enabled,
            "MODE": mode,
            "CORRIDOR": corridor,
            "POLYGON": polygon,
        }

    def _build_metric_grid(self, H, W, device):
        pc = self._get_point_cloud_range()
        x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc]
        dx = (x_max - x_min) / float(W)
        dy = (y_max - y_min) / float(H)
        xs = x_min + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * dx
        ys = y_min + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * dy
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return xx, yy, (x_min, y_min, x_max, y_max)

    def _polygon_mask_hw(self, H, W, pc_range_xy, polygon_xy, device):
        x_min, y_min, x_max, y_max = pc_range_xy
        if len(polygon_xy) < 3:
            return torch.zeros((H, W), dtype=torch.float32, device=device)

        dx = (x_max - x_min) / float(W)
        dy = (y_max - y_min) / float(H)

        pts = []
        for x, y in polygon_xy:
            ix = int(np.round((x - x_min) / dx - 0.5))
            iy = int(np.round((y - y_min) / dy - 0.5))
            ix = int(np.clip(ix, 0, W - 1))
            iy = int(np.clip(iy, 0, H - 1))
            pts.append([ix, iy])

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 1)
        return torch.from_numpy(mask).to(device=device, dtype=torch.float32)

    def _build_roi_map(self, xx, yy, H, W, pc_range_xy, roi_cfg):
        if (roi_cfg is None) or (not roi_cfg.get("ENABLED", True)):
            return torch.ones((H, W), dtype=torch.float32, device=xx.device)

        mode = str(roi_cfg.get("MODE", "corridor")).lower()
        if mode == "polygon":
            poly = roi_cfg.get("POLYGON", [])
            return self._polygon_mask_hw(H, W, pc_range_xy, poly, xx.device)

        c = roi_cfg.get("CORRIDOR", {})
        k = float(c.get("K", 1.53))
        b = float(c.get("B", 18.5))
        x0 = float(c.get("X0", -80.0))
        x1 = float(c.get("X1", 90.0))
        half_left = float(c.get("HALF_LEFT", 12.0))
        half_right = float(c.get("HALF_RIGHT", 6.0))
        den = (k * k + 1.0) ** 0.5
        dist_signed = (k * xx - yy + b) / den
        return ((xx > x0) & (xx < x1) & (dist_signed > -half_left) & (dist_signed < half_right)).float()

    @staticmethod
    def _points_in_polygon(points_xy, polygon_xy):
        if len(polygon_xy) < 3:
            return np.zeros((points_xy.shape[0],), dtype=bool)
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        poly = np.asarray(polygon_xy, dtype=np.float32)
        inside = np.zeros(points_xy.shape[0], dtype=bool)
        j = poly.shape[0] - 1
        for i in range(poly.shape[0]):
            xi, yi = poly[i, 0], poly[i, 1]
            xj, yj = poly[j, 0], poly[j, 1]
            inter = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside ^= inter
            j = i
        return inside

    def _is_inside_track_roi(self, centers_xy):
        cfg = getattr(self, "track_roi_cfg", None)
        if (cfg is None) or (not cfg.get("ENABLED", True)):
            return np.ones((centers_xy.shape[0],), dtype=bool)

        mode = str(cfg.get("MODE", "corridor")).lower()
        if mode == "polygon":
            return self._points_in_polygon(centers_xy, cfg.get("POLYGON", []))

        c = cfg.get("CORRIDOR", {})
        k = float(c.get("K", 1.53))
        b = float(c.get("B", 18.5))
        x0 = float(c.get("X0", -80.0))
        x1 = float(c.get("X1", 90.0))
        half_left = float(c.get("HALF_LEFT", 9.0))
        half_right = float(c.get("HALF_RIGHT", 3.5))

        x = centers_xy[:, 0]
        y = centers_xy[:, 1]
        den = (k * k + 1.0) ** 0.5
        d = (k * x - y + b) / den
        return (x > x0) & (x < x1) & (d > -half_left) & (d < half_right)

    def _filter_pred_by_track_roi(self, pred_dicts):
        cfg = getattr(self, "track_roi_cfg", None)
        if (cfg is None) or (not cfg.get("ENABLED", True)):
            return pred_dicts

        out = []
        for pd in pred_dicts:
            if "pred_boxes" not in pd:
                out.append(pd)
                continue

            boxes = pd["pred_boxes"]
            if torch.is_tensor(boxes):
                if boxes.numel() == 0:
                    out.append(pd)
                    continue
                centers = boxes[:, :2].detach().cpu().numpy()
                keep_np = self._is_inside_track_roi(centers)
                keep = torch.from_numpy(keep_np).to(device=boxes.device, dtype=torch.bool)
                n = boxes.shape[0]
            else:
                boxes_np = np.asarray(boxes)
                if boxes_np.size == 0:
                    out.append(pd)
                    continue
                centers = boxes_np[:, :2]
                keep_np = self._is_inside_track_roi(centers)
                keep = keep_np
                n = boxes_np.shape[0]

            pd2 = {}
            for k, v in pd.items():
                if torch.is_tensor(v) and v.shape[0] == n:
                    pd2[k] = v[keep]
                elif isinstance(v, np.ndarray) and v.shape[0] == n:
                    pd2[k] = v[keep_np]
                else:
                    pd2[k] = v
            out.append(pd2)
        return out

    def _get_point_cloud_range(self):
        """
        Robust POINT_CLOUD_RANGE getter.
        Priority:
        1) self.point_cloud_range (fusion init 写入的)
        2) MODEL.IMAGE_TO_BEV.POINT_CLOUD_RANGE (yaml 显式写的)
        3) dataset_cfg.POINT_CLOUD_RANGE (dataset yaml 里写的)
        """
        # 1) 融合 init 写过的（你 __init__ 里有 self.point_cloud_range = itb_cfg['POINT_CLOUD_RANGE']）
        if hasattr(self, "point_cloud_range") and (self.point_cloud_range is not None):
            return self.point_cloud_range

        # 2) yaml 里显式写的
        itb = getattr(self.model_cfg, "IMAGE_TO_BEV", None)
        if itb is not None and hasattr(itb, "POINT_CLOUD_RANGE"):
            return itb.POINT_CLOUD_RANGE

        # 3) dataset_cfg 里写的（可能是 EasyDict 或 dict）
        dcfg = getattr(self, "dataset_cfg", None)
        if dcfg is not None:
            if isinstance(dcfg, dict) and ("POINT_CLOUD_RANGE" in dcfg):
                return dcfg["POINT_CLOUD_RANGE"]
            if hasattr(dcfg, "POINT_CLOUD_RANGE"):
                return dcfg.POINT_CLOUD_RANGE

        raise KeyError("POINT_CLOUD_RANGE not found in (self.point_cloud_range / model_cfg.IMAGE_TO_BEV / dataset_cfg).")


    def apply_freeze(self):
        """
        Unified freeze controller.
        Modes:
        - stage1
        - stage2_head_only
        - stage2_unfreeze_backbone
        - calib_xy
        - none
        """
        # 先全部冻结（最稳）
        for p in self.parameters():
            p.requires_grad = False

        mode = getattr(self.model_cfg, "FREEZE_MODE", "none")
        print(f"[FREEZE] mode = {mode}")

        def freeze_module(name):
            if hasattr(self, name) and getattr(self, name) is not None:
                for p in getattr(self, name).parameters():
                    p.requires_grad = False

        def unfreeze_module(name):
            if hasattr(self, name) and getattr(self, name) is not None:
                for p in getattr(self, name).parameters():
                    p.requires_grad = True

        if mode == "stage1":
            for n in ['img_backbone', 'img_to_bev', 'bev_fusion']:
                unfreeze_module(n)

        elif mode == "stage2_head_only":
            for n in ['dense_head', 'bev_fusion']:
                unfreeze_module(n)
            for n in ['img_backbone', 'img_to_bev']:
                freeze_module(n)

        elif mode == "stage2_unfreeze_backbone":
            for n in ['backbone_2d', 'dense_head', 'bev_fusion']:
                unfreeze_module(n)
            for n in ['img_backbone', 'img_to_bev']:
                freeze_module(n)

        elif mode == "calib_xy":
            # 如果没创建 calib_refiner，就别让它 0 参数白训崩溃
            if hasattr(self, "calib_refiner") and (self.calib_refiner is not None):
                unfreeze_module("calib_refiner")
            else:
                print("[FREEZE][WARN] calib_xy but calib_refiner is None. Fallback to unfreeze bev_fusion to avoid 0-trainable.")
                unfreeze_module("bev_fusion")

            for n in ['img_backbone', 'img_to_bev', 'bev_fusion', 'dense_head', 'backbone_2d', 'vfe', 'map_to_bev_module']:
                freeze_module(n)
            # 若 fallback 解冻了 bev_fusion，这里再打开（保持一致）
            if (not hasattr(self, "calib_refiner")) or (self.calib_refiner is None):
                unfreeze_module("bev_fusion")

        elif mode == "none":
            for p in self.parameters():
                p.requires_grad = True

        n_train = sum(int(p.requires_grad) for p in self.parameters())
        n_all = sum(1 for _ in self.parameters())
        print(f"[FREEZE] trainable params: {n_train}/{n_all}")

        def _count_trainable(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        # 这些模块不一定都存在，做个保护
        if hasattr(self, "img_backbone"):
            print("[FREEZE-CHECK] img_backbone", _count_trainable(self.img_backbone))
        if hasattr(self, "img_to_bev"):
            print("[FREEZE-CHECK] img_to_bev", _count_trainable(self.img_to_bev))
        if hasattr(self, "bev_fusion"):
            print("[FREEZE-CHECK] bev_fusion", _count_trainable(self.bev_fusion))
        if hasattr(self, "dense_head"):
            print("[FREEZE-CHECK] dense_head", _count_trainable(self.dense_head))
        if hasattr(self, "backbone_2d"):
            print("[FREEZE-CHECK] backbone_2d", _count_trainable(self.backbone_2d))
        if hasattr(self, "vfe") and hasattr(self, "map_to_bev_module"):
            print("[FREEZE-CHECK] vfe/scatter", _count_trainable(self.vfe) + _count_trainable(self.map_to_bev_module))
        if hasattr(self, "calib_refiner") and self.calib_refiner is not None:
            print("[FREEZE-CHECK] calib_refiner", _count_trainable(self.calib_refiner))

    def _sobel_mag(self, x):
        gx = F.conv2d(x, self._sobel_kx.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self._sobel_ky.to(dtype=x.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _boxes3d_corners_lidar(self, boxes3d_np):
        M = boxes3d_np.shape[0]
        x, y, z, dx, dy, dz, yaw = [boxes3d_np[:, i] for i in range(7)]

        x_c = np.stack([dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2], axis=1)
        y_c = np.stack([dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2], axis=1)
        z_c = np.stack([dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2], axis=1)

        corners = np.stack([x_c, y_c, z_c], axis=2).astype(np.float32)

        cos, sin = np.cos(yaw), np.sin(yaw)
        R = np.zeros((M, 3, 3), dtype=np.float32)
        R[:, 0, 0] = cos
        R[:, 0, 1] = -sin
        R[:, 1, 0] = sin
        R[:, 1, 1] = cos
        R[:, 2, 2] = 1.0

        corners = corners @ np.transpose(R, (0, 2, 1))
        corners[:, :, 0] += x[:, None]
        corners[:, :, 1] += y[:, None]
        corners[:, :, 2] += z[:, None]
        return corners

    def _project_points(self, X_lidar, K, T_cam_from_lidar):
        pts = np.asarray(X_lidar, dtype=np.float32).reshape(-1, 3)
        pts_h = np.concatenate(
            [pts, np.ones((pts.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        pts_cam = (np.asarray(T_cam_from_lidar, dtype=np.float32) @ pts_h.T).T[:, :3]
        depth = pts_cam[:, 2].astype(np.float32)
        depth_safe = np.where(depth > 1e-6, depth, 1e-6).astype(np.float32)

        u = K[0, 0] * (pts_cam[:, 0] / depth_safe) + K[0, 2]
        v = K[1, 1] * (pts_cam[:, 1] / depth_safe) + K[1, 2]
        pts2d = np.stack([u, v], axis=1).astype(np.float32)
        return pts2d, depth

    def _draw_projected_box(self, img_bgr, pts2d, color=(0, 255, 0), thickness=2):
        pts = pts2d.astype(np.int32)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for i, j in edges:
            cv2.line(img_bgr, tuple(pts[i]), tuple(pts[j]), color, thickness, lineType=cv2.LINE_AA)
    def _tweak_calib_torch(self, cam_K, T_cam_from_lidar, image_W, image_H, img_tensor):
        """
        在“相机坐标系”对外参做一个微小可调修正（不改训练，只用于快速对齐/调参）。
        同时可按实际输入图尺寸对 K 做缩放（防止 K 与 resize 后图像不一致）。
        - cam_K: (B,3,3) torch
        - T_cam_from_lidar: (B,4,4) torch
        - image_W/image_H: (B,) torch（你的 dataset 里是固定 1920/1080）
        - img_tensor: (B,3,H,W) torch（实际网络输入）
        """
        import os
        import torch
        import numpy as np

        device = cam_K.device
        dtype = cam_K.dtype
        B = cam_K.shape[0]

        # ---------- 1) K 按 resize 缩放（一般你这里 sx=sy=1，但留着更稳） ----------
        cam_K2 = cam_K.clone()
        if (image_W is not None) and (image_H is not None):
            H_img = float(img_tensor.shape[-2])
            W_img = float(img_tensor.shape[-1])
            sx = (W_img / (image_W.to(dtype=dtype) + 1e-6))  # (B,)
            sy = (H_img / (image_H.to(dtype=dtype) + 1e-6))  # (B,)

            cam_K2[:, 0, 0] = cam_K2[:, 0, 0] * sx
            cam_K2[:, 0, 2] = cam_K2[:, 0, 2] * sx
            cam_K2[:, 1, 1] = cam_K2[:, 1, 1] * sy
            cam_K2[:, 1, 2] = cam_K2[:, 1, 2] * sy

        # ---------- 2) 应用历史运行时外参微调 ----------
        dpitch = float(os.environ.get("CALIB_DPITCH_DEG", "0"))
        droll  = float(os.environ.get("CALIB_DROLL_DEG",  "0"))
        dyaw   = float(os.environ.get("CALIB_DYAW_DEG",   "0"))
        dx_cam = float(os.environ.get("CALIB_DX_CAM", "0"))
        dy_cam = float(os.environ.get("CALIB_DY_CAM", "0"))
        dz_cam = float(os.environ.get("CALIB_DZ_CAM", "0"))

        if (abs(dpitch) + abs(droll) + abs(dyaw) + abs(dx_cam) + abs(dy_cam) + abs(dz_cam)) > 1e-12:
            def _rotx(a):
                c, s = np.cos(a), np.sin(a)
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

            def _roty(a):
                c, s = np.cos(a), np.sin(a)
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            def _rotz(a):
                c, s = np.cos(a), np.sin(a)
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            aP = np.deg2rad(dpitch)
            aR = np.deg2rad(droll)
            aY = np.deg2rad(dyaw)
            T_delta = np.eye(4, dtype=np.float32)
            T_delta[:3, :3] = (_rotz(aY) @ _roty(aP) @ _rotx(aR)).astype(np.float32)
            T_delta[:3, 3] = np.array([dx_cam, dy_cam, dz_cam], dtype=np.float32)
            T_delta = torch.as_tensor(T_delta, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            T_cam_from_lidar = torch.matmul(T_delta, T_cam_from_lidar)
        return cam_K2, T_cam_from_lidar

    def forward(self, batch_dict):
        import os, cv2

        if not hasattr(self, '_printed_once'):
            self._printed_once = True
            print('batch_dict keys:', sorted(batch_dict.keys()))
            if 'images' in batch_dict:
                print('images:', batch_dict['images'].shape, batch_dict['images'].dtype)
            if 'cam_K' in batch_dict:
                print('cam_K:', batch_dict['cam_K'].shape, batch_dict['cam_K'].dtype)
            if 'T_cam_from_lidar' in batch_dict:
                print('T_cam_from_lidar:', batch_dict['T_cam_from_lidar'].shape, batch_dict['T_cam_from_lidar'].dtype)

        fused_once = False

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            if not hasattr(self, '_dbg_modules'):
                self._dbg_modules = True
                print(">>> module_list:", [m.__class__.__name__ for m in self.module_list])

            if (self.use_fusion and (not fused_once)
                and ('spatial_features' in batch_dict)
                and ('images' in batch_dict)
                and ('cam_K' in batch_dict)
                and ('T_cam_from_lidar' in batch_dict)
                and (cur_module.__class__.__name__ == "PointPillarScatter")
            ):
                # 每次进入融合点，先清空（避免上一 batch 残留）
                self._loss_calib = None
                self._tb_calib = {}

                lidar_bev = batch_dict['spatial_features']
                img = batch_dict['images'].to(lidar_bev.device)

                # 1) 图像特征
                img_feat = self.img_backbone(img)

                # 2) 标定信息（按 ImageToBEV 的接口）
                if ('image_H' in batch_dict) and ('image_W' in batch_dict):
                    image_hw = torch.stack([batch_dict['image_H'], batch_dict['image_W']], dim=1)  # (B,2)
                else:
                    image_hw = None

                # === 用可调外参/可缩放K 的 calib（让融合真正变准）===
                cam_K = batch_dict['cam_K']
                T_cam = batch_dict['T_cam_from_lidar']

                # img 是 (B,3,H,W) 的网络输入
                cam_K, T_cam = self._tweak_calib_torch(
                    cam_K, T_cam,
                    batch_dict.get('image_W', None),
                    batch_dict.get('image_H', None),
                    img
                )

                calib = {
                    'cam_K': cam_K,
                    'T_cam_from_lidar': T_cam,
                    'image_hw': image_hw
                }


                # 3) Image -> BEV：先投影一次
                img_feat_proj = img_feat.detach() if getattr(self, "detach_img_feat_for_proj", False) else img_feat
                img_bev0 = self.img_to_bev(img_feat_proj, lidar_bev, calib)

                delta_xy = None
                valid_mask = None

                # 4) 预测 Δx/Δy 并二次投影（拿 valid_mask 回来做 mask）
                if getattr(self, "use_calib_refine", False) and (self.calib_refiner is not None):
                    lidar_in = lidar_bev.detach()
                    img0_in = img_bev0.detach() if getattr(self, "detach_img0_for_refiner", True) else img_bev0
                    delta_xy = self.calib_refiner(lidar_in, img0_in)  # (B,2)
                    if self.training:
                        if not hasattr(self, "_dbg_iter"):
                            self._dbg_iter = 0
                        self._dbg_iter += 1
                        if (self._dbg_iter % 20) == 0:  # 每20步打印一次
                            dx = float(delta_xy[:, 0].mean().detach().cpu())
                            dy = float(delta_xy[:, 1].mean().detach().cpu())
                            print(f"[CALIB_XY] iter={self._dbg_iter} dx_mean={dx:.4f} dy_mean={dy:.4f}")


                    calib2 = dict(calib)
                    calib2["delta_xy_lidar"] = delta_xy
                    calib2["return_valid"] = True

                    out = self.img_to_bev(img_feat_proj, lidar_bev, calib2)
                    if isinstance(out, (tuple, list)) and len(out) == 2:
                        img_bev, valid_mask = out
                    else:
                        img_bev, valid_mask = out, None
                else:
                    img_bev = img_bev0

                # ===== ROI mask（支持 corridor/polygon 双模式） =====
                B, _, H, W = lidar_bev.shape
                device = img_bev.device
                xx, yy, pc_xy = self._build_metric_grid(H, W, device)

                fusion_roi_map = self._build_roi_map(xx, yy, H, W, pc_xy, self.fusion_roi_cfg)
                track_roi_map = self._build_roi_map(xx, yy, H, W, pc_xy, self.track_roi_cfg)

                fusion_roi_mask = fusion_roi_map[None, None, :, :].repeat(B, 1, 1, 1)

                if not hasattr(self, "_dbg_shape_once"):
                    self._dbg_shape_once = True
                    print("[FUSION SHAPE] lidar_bev", tuple(lidar_bev.shape),
                          "img_bev", tuple(img_bev.shape),
                          "fusion_roi_mask", tuple(fusion_roi_mask.shape))

                if (os.environ.get("SAVE_FUSION_DEBUG", "0") == "1") and (not hasattr(self, "_saved_road_mask")):
                    os.makedirs("tools/output", exist_ok=True)
                    m = (fusion_roi_map.detach().cpu().numpy() * 255).astype("uint8")
                    ok = cv2.imwrite("tools/output/road_mask.png", m)
                    m2 = (track_roi_map.detach().cpu().numpy() * 255).astype("uint8")
                    ok2 = cv2.imwrite("tools/output/track_mask.png", m2)
                    print(
                        f"[ROAD] saved masks fusion_ok={ok} track_ok={ok2} "
                        f"fusion_mean={float(fusion_roi_map.mean()):.4f} track_mean={float(track_roi_map.mean()):.4f}"
                    )
                    self._saved_road_mask = True

                # ===== ROI 内结构对齐 loss：驱动 delta_xy（只在训练时）=====
                if self.training and (delta_xy is not None) and getattr(self, "use_calib_refine", False):
                    lidar_src = lidar_bev.detach() if getattr(self, "detach_lidar_for_align", True) else lidar_bev
                    img_src = img_bev  # 这里必须保留梯度，才能反传到 delta_xy

                    # 单通道结构响应
                    lidar_map = lidar_src.abs().mean(dim=1, keepdim=True)
                    img_map = img_src.abs().mean(dim=1, keepdim=True)

                    # 归一化到 0~1（避免尺度飘）
                    def _norm01(m_):
                        mn = m_.amin(dim=(2, 3), keepdim=True)
                        mx = m_.amax(dim=(2, 3), keepdim=True)
                        return (m_ - mn) / (mx - mn + 1e-6)

                    lidar_map = _norm01(lidar_map)
                    img_map = _norm01(img_map)

                    edge_l = self._sobel_mag(lidar_map)
                    edge_i = self._sobel_mag(img_map)

                    mask = fusion_roi_mask
                    if getattr(self, "use_valid_mask", True) and (valid_mask is not None):
                        mask = mask * valid_mask

                    denom = mask.sum().clamp(min=1.0)
                    loss_align = ((edge_l - edge_i).abs() * mask).sum() / denom
                    loss_reg = (delta_xy ** 2).sum(dim=1).mean()

                    self._loss_calib = self.loss_align_w * loss_align + self.loss_reg_w * loss_reg
                    with torch.no_grad():
                        self._tb_calib = {
                            "loss_align": float(loss_align.detach().cpu()),
                            "loss_reg": float(loss_reg.detach().cpu()),
                            "dx_mean": float(delta_xy[:, 0].mean().detach().cpu()),
                            "dy_mean": float(delta_xy[:, 1].mean().detach().cpu()),
                        }

                # 融合（ROI 约束统一由 BEVFusion 内部处理，不在外部重复乘 mask）
                fused = self.bev_fusion(lidar_bev, img_bev, roi_mask=fusion_roi_mask)

                if not hasattr(self, "_dbg_once"):
                    self._dbg_once = True
                    a = self.bev_fusion.alpha if hasattr(self.bev_fusion, "alpha") else None
                    if a is not None:
                        print("[BEV_FUSION] alpha=", float(a.detach().cpu()))
                    print("[BEV_FUSION] lidar_mean=", float(lidar_bev.abs().mean().detach().cpu()))
                    print("[BEV_FUSION] img_mean=", float(img_bev.abs().mean().detach().cpu()))
                    print("[BEV_FUSION] fused_minus_lidar_mean=", float((fused - lidar_bev).abs().mean().detach().cpu()))
                    print("[BEV_FUSION] roi_mean=", float(fusion_roi_mask.mean().detach().cpu()))

                batch_dict['spatial_features'] = fused

                # debug：保存一次 img_bev heatmap
                if (os.environ.get("SAVE_FUSION_DEBUG", "0") == "1") and (not hasattr(self, '_dbg_saved_heatmap')):
                    self._dbg_saved_heatmap = True
                    try:
                        os.makedirs('tools/output', exist_ok=True)
                        hm = img_bev[0].mean(dim=0).detach().float().cpu().numpy()
                        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
                        out = (hm * 255).astype('uint8')
                        cv2.imwrite('tools/output/img_bev_heatmap.png', out)
                        print('[FUSION] saved heatmap -> tools/output/img_bev_heatmap.png')
                    except Exception as e:
                        print('[FUSION] heatmap save failed:', repr(e))

                fused_once = True

        if self.training:
            mode = getattr(self.model_cfg, "FREEZE_MODE", "none")

            # calib_xy：只训练 calib loss（大幅省显存，避免 OOM）
            if mode == "calib_xy" and getattr(self, "_loss_calib", None) is not None:
                tb_dict = {
                    "loss_calib": float(self._loss_calib.detach().cpu())
                }
                if getattr(self, "_tb_calib", None):
                    for k, v in self._tb_calib.items():
                        tb_dict[f"calib/{k}"] = v

                ret_dict = {"loss": self._loss_calib}
                return ret_dict, tb_dict, {}

            # 其它模式：正常训练检测 loss（rpn/head）
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            pred_dicts = self._filter_pred_by_track_roi(pred_dicts)

            # ====== Save ONE overlay image (only once) ======
            if not hasattr(self, "_saved_overlay_once"):
                self._saved_overlay_once = True
                try:
                    need_keys = ("images", "cam_K", "T_cam_from_lidar", "frame_id", "image_H", "image_W")
                    if not all(k in batch_dict for k in need_keys):
                        print("[Overlay] missing keys:", [k for k in need_keys if k not in batch_dict])
                        return pred_dicts, recall_dicts

                    if len(pred_dicts) == 0 or ("pred_boxes" not in pred_dicts[0]) or pred_dicts[0]["pred_boxes"].shape[0] == 0:
                        print("[Overlay] no pred boxes")
                        return pred_dicts, recall_dicts

                    fid_raw = batch_dict["frame_id"][0]
                    fid = str(fid_raw)
                    fid_clean = fid.replace("\\", "/")
                    if "/" in fid_clean:
                        seq, fr = fid_clean.split("/")[-2], fid_clean.split("/")[-1]
                    else:
                        seq, fr = "1802", fid_clean
                    fr = fr.zfill(6)
                    frame_tag = f"{seq}_{fr}"

                    img_t = batch_dict["images"][0].detach().cpu()
                    img = img_t.permute(1, 2, 0).contiguous().numpy()

                    if img.dtype != np.uint8:
                        mx = float(img.max())
                        if mx <= 1.5:
                            img = (img * 255.0).round()
                        img = np.clip(img, 0, 255).astype(np.uint8)

                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    H_img, W_img = img_bgr.shape[:2]

                    H0 = int(batch_dict["image_H"][0])
                    W0 = int(batch_dict["image_W"][0])

                    K0 = batch_dict["cam_K"][0].detach().cpu().numpy().astype(np.float32)
                    T = batch_dict["T_cam_from_lidar"][0].detach().cpu().numpy().astype(np.float32)

                    # ---------- overlay 用的 K/T 修正 + 过滤 ----------
                    K = K0.copy()

                    # 1) K 按图像缩放修正（一般 sx=sy=1，但留着更稳）
                    sx = float(W_img) / float(W0 + 1e-6)
                    sy = float(H_img) / float(H0 + 1e-6)
                    if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
                        K[0, 0] *= sx
                        K[0, 2] *= sx
                        K[1, 1] *= sy
                        K[1, 2] *= sy

                    # 2) 外参微调旋钮（相机坐标系）
                    def _rotx(a):
                        c, s = np.cos(a), np.sin(a)
                        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

                    def _roty(a):
                        c, s = np.cos(a), np.sin(a)
                        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

                    def _rotz(a):
                        c, s = np.cos(a), np.sin(a)
                        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

                    dpitch = float(os.environ.get("CALIB_DPITCH_DEG", "0"))
                    droll  = float(os.environ.get("CALIB_DROLL_DEG",  "0"))
                    dyaw   = float(os.environ.get("CALIB_DYAW_DEG",   "0"))
                    dx_cam = float(os.environ.get("CALIB_DX_CAM", "0"))
                    dy_cam = float(os.environ.get("CALIB_DY_CAM", "0"))
                    dz_cam = float(os.environ.get("CALIB_DZ_CAM", "0"))

                    if (abs(dpitch) + abs(droll) + abs(dyaw) + abs(dx_cam) + abs(dy_cam) + abs(dz_cam)) > 1e-9:
                        aP = np.deg2rad(dpitch)
                        aR = np.deg2rad(droll)
                        aY = np.deg2rad(dyaw)
                        T_delta = np.eye(4, dtype=np.float32)
                        T_delta[:3, :3] = (_rotz(aY) @ _roty(aP) @ _rotx(aR)).astype(np.float32)
                        T_delta[:3, 3] = np.array([dx_cam, dy_cam, dz_cam], dtype=np.float32)
                        T = (T_delta @ T).astype(np.float32)

                    print(
                        f"[Overlay] fid={fid_clean} rawWH=({W0},{H0}) imgWH=({W_img},{H_img}) scale=({sx:.4f},{sy:.4f}) "
                        f"dpitch={dpitch} droll={droll} dyaw={dyaw} dxyz=({dx_cam},{dy_cam},{dz_cam})"
                    )

                    # 3) 过滤：去掉“离谱大框/角点穿越导致扭曲”的情况（非常关键）
                    score_th  = float(os.environ.get("OVERLAY_SCORE_TH", "0.30"))
                    depth_min = float(os.environ.get("OVERLAY_DEPTH_MIN", "1.0"))

                    dx_max = float(os.environ.get("OVERLAY_DX_MAX", "20.0"))
                    dy_max = float(os.environ.get("OVERLAY_DY_MAX", "8.0"))
                    dz_max = float(os.environ.get("OVERLAY_DZ_MAX", "6.0"))
                    dx_min = float(os.environ.get("OVERLAY_DX_MIN", "1.0"))
                    dy_min = float(os.environ.get("OVERLAY_DY_MIN", "0.8"))
                    dz_min = float(os.environ.get("OVERLAY_DZ_MIN", "0.8"))

                    boxes = pred_dicts[0]["pred_boxes"].detach().cpu().numpy().astype(np.float32)
                    scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy().astype(np.float32)
                    corners = self._boxes3d_corners_lidar(boxes)

                    for i in range(corners.shape[0]):
                        if scores[i] < score_th:
                            continue

                        # 尺寸 sanity：砍掉右下角那种巨大斜框
                        dx, dy, dz = float(boxes[i, 3]), float(boxes[i, 4]), float(boxes[i, 5])
                        if (dx > dx_max) or (dy > dy_max) or (dz > dz_max) or (dx < dx_min) or (dy < dy_min) or (dz < dz_min):
                            continue

                        pts2d, depth = self._project_points(corners[i], K, T)
                        if np.any(np.isnan(pts2d)) or np.any(np.isinf(pts2d)):
                            continue

                        # 严格深度：8/8 角点都必须在相机前方，避免“扭曲倾斜假象”
                        if float(np.min(depth)) <= depth_min:
                            continue

                        inside = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < W_img) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < H_img)
                        if int(inside.sum()) < 6:
                            continue

                        # 2D 外接框面积过滤：覆盖太大通常是 warped FP
                        x1, y1 = float(np.min(pts2d[:, 0])), float(np.min(pts2d[:, 1]))
                        x2, y2 = float(np.max(pts2d[:, 0])), float(np.max(pts2d[:, 1]))
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        w2d = max(0.0, x2 - x1)
                        h2d = max(0.0, y2 - y1)

                        # 过滤“过长/过高”的投影框（专杀右下角这种长条）
                        if (w2d > 0.45 * W_img) or (h2d > 0.55 * H_img):
                            continue
                        if area > 0.12 * float(W_img * H_img):
                            continue

                        self._draw_projected_box(img_bgr, pts2d, color=(0, 255, 0), thickness=2)
                        p0 = pts2d[0].astype(int)
                        p0[0] = int(np.clip(p0[0], 0, W_img - 1))
                        p0[1] = int(np.clip(p0[1], 0, H_img - 1))
                        cv2.putText(img_bgr, f"{scores[i]:.2f}", tuple(p0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    tag = os.environ.get("EVAL_TAG", "run")
                    out_dir = f"tools/output/overlay_{seq}_{tag}"

                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{frame_tag}.jpg")
                    cv2.imwrite(out_path, img_bgr)
                    print(f"[Saved overlay] {out_path}")

                except Exception as e:
                    print("[Overlay error]", e)

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}

        loss = loss_rpn

        # Optional regularization: avoid fusion alpha collapsing to near-zero
        if getattr(self, "use_fusion", False) and hasattr(self, "bev_fusion"):
            bf_cfg = getattr(self.model_cfg, "BEV_FUSION", None) or {}
            alpha_reg_w = float(self._cfg_get(bf_cfg, "ALPHA_REG_W", 0.0))
            if alpha_reg_w > 0:
                alpha_target = float(self._cfg_get(bf_cfg, "ALPHA_TARGET_MIN", 0.03))
                alpha_now = torch.clamp(self.bev_fusion.alpha, min=0.0)
                loss_alpha_reg = F.relu(alpha_target - alpha_now).pow(2)
                loss = loss + alpha_reg_w * loss_alpha_reg
                tb_dict['loss_alpha_reg'] = float(loss_alpha_reg.detach().cpu())
                tb_dict['alpha_now'] = float(alpha_now.detach().cpu())

        # 把 calib loss 加进去
        if getattr(self, "_loss_calib", None) is not None:
            loss = loss + self._loss_calib
            tb_dict['loss_calib'] = float(self._loss_calib.detach().cpu())
            if getattr(self, "_tb_calib", None):
                for k, v in self._tb_calib.items():
                    tb_dict[f'calib/{k}'] = v

        return loss, tb_dict, disp_dict
