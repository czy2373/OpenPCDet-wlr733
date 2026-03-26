import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _as_hw(image_hw):
    """
    Normalize image_hw into (H, W) python ints.
    Accepts:
      - tuple/list: (H, W)
      - torch.Tensor: shape (2,) or (B,2)
    """
    if image_hw is None:
        return 1080, 1920

    # tensor
    if torch.is_tensor(image_hw):
        if image_hw.numel() == 2:
            h = int(image_hw.view(-1)[0].item())
            w = int(image_hw.view(-1)[1].item())
            return h, w
        # (B,2) -> take first
        h = int(image_hw[0, 0].item())
        w = int(image_hw[0, 1].item())
        return h, w

    # list/tuple
    if isinstance(image_hw, (list, tuple)) and len(image_hw) == 2:
        return int(image_hw[0]), int(image_hw[1])

    raise TypeError(f"Unsupported image_hw type/shape: {type(image_hw)}")

class ImageToBEV(nn.Module):
    """
    Image feature -> BEV feature by projecting BEV grid centers (defined strictly by lidar_bev H/W)
    into image feature map using K and T_cam_from_lidar.

    STRICT RULE:
      - BEV grid coordinates are derived from lidar_bev shape + (POINT_CLOUD_RANGE, VOXEL_SIZE).
      - Never use anchor_range to define BEV grid.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # required: voxel size used by voxelization (xy)
        vs = model_cfg.get('VOXEL_SIZE', [0.2, 0.2, 10])
        self.vx = float(vs[0])
        self.vy = float(vs[1])

        # required: must match dataset POINT_CLOUD_RANGE
        pc_range = model_cfg.get('POINT_CLOUD_RANGE', None)
        if pc_range is None:
            # keep a safe default but strongly建议你在 yaml 里显式配置
            pc_range = [-112.0, -106.0, -12.0, 99.2, 158.0, 2.0]
        self.x_min = float(pc_range[0])
        self.y_min = float(pc_range[1])

        # image backbone stride (feature map downsample vs resized image)
        self.img_downsample = int(model_cfg.get('IMG_DOWNSAMPLE', 4))

        # BEV projection downsample (compute on coarse grid then upsample to lidar_bev)
        self.bev_downsample = int(model_cfg.get('BEV_DOWNSAMPLE', 4))

        # projection planes
        self.z0 = float(model_cfg.get('Z0', -2.0))
        z_list = model_cfg.get('Z_LIST', [])
        if z_list is None:
            z_list = []
        if len(z_list) == 0:
            self.z_list = [self.z0]
        else:
            self.z_list = [float(z) for z in z_list]

        self.debug_save = bool(model_cfg.get('DEBUG_SAVE', False))

    @torch.no_grad()
    def _save_debug_heatmap(self, bev_feat, path):
        # bev_feat: (B,C,H,W)
        x = bev_feat[0].detach().float()
        x = x.abs().mean(dim=0)  # (H,W)
        x = x - x.min()
        if x.max() > 1e-6:
            x = x / x.max()
        x = (x * 255.0).clamp(0, 255).byte().cpu().numpy()
        import cv2
        cv2.imwrite(path, x)

    def _make_bev_centers_xy(self, lidar_bev, device):
        """
        Create BEV grid centers on a COARSE grid (downsampled), strictly aligned with lidar_bev size.
        Return:
          xs: (Wc,)
          ys: (Hc,)
          Hc, Wc
        """
        B, C, H, W = lidar_bev.shape
        ds = max(1, int(self.bev_downsample))

        Hc = (H + ds - 1) // ds
        Wc = (W + ds - 1) // ds

        off = (ds / 2.0) - 0.5
        iy = torch.arange(Hc, device=device, dtype=torch.float32) * ds + off
        ix = torch.arange(Wc, device=device, dtype=torch.float32) * ds + off

        iy = iy.clamp(0, H - 1)
        ix = ix.clamp(0, W - 1)

        if Hc <= 0 or Wc <= 0:
            raise ValueError(f"Invalid lidar_bev shape {lidar_bev.shape} with BEV_DOWNSAMPLE={ds}")

        # pick centers of each ds x ds block: idx = k*ds + (ds/2 - 0.5)
        off = (ds / 2.0) - 0.5
        iy = torch.arange(Hc, device=device, dtype=torch.float32) * ds + off
        ix = torch.arange(Wc, device=device, dtype=torch.float32) * ds + off

        # convert index -> metric (x,y) using SAME pc_range & voxel_size as lidar bev
        xs = self.x_min + (ix + 0.5) * self.vx   # (Wc,)
        ys = self.y_min + (iy + 0.5) * self.vy   # (Hc,)
        return xs, ys, Hc, Wc

    def _project_points(self, xs, ys, z, cam_K, T_cam_from_lidar, delta_xy=None):
        """
        xs: (Wc,), ys: (Hc,)
        return u,v,valid in resized-image pixel coordinates (not feature coords yet)
        Shapes: u,v,valid -> (B,Hc,Wc)
        """
        batch_size = int(delta_xy.shape[0]) if delta_xy is not None else int(cam_K.shape[0])
        dtype = cam_K.dtype
        Hc = int(ys.shape[0])
        Wc = int(xs.shape[0])

        yy, xx = torch.meshgrid(
            ys.to(dtype=dtype),
            xs.to(dtype=dtype),
            indexing='ij',
        )
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)
        zz = torch.full_like(xx, float(z))

        if delta_xy is not None:
            xx = xx + delta_xy[:, 0].view(batch_size, 1, 1).to(dtype=dtype)
            yy = yy + delta_xy[:, 1].view(batch_size, 1, 1).to(dtype=dtype)

        ones = torch.ones_like(xx)
        pts = torch.stack([xx, yy, zz, ones], dim=-1).view(batch_size, -1, 4).transpose(1, 2)
        pts_cam = torch.matmul(T_cam_from_lidar.to(dtype=dtype), pts)

        X = pts_cam[:, 0, :]
        Y = pts_cam[:, 1, :]
        Z = pts_cam[:, 2, :]
        valid = Z > 1e-3
        Z_safe = Z.clamp(min=1e-3)

        fx = cam_K[:, 0, 0].unsqueeze(1)
        fy = cam_K[:, 1, 1].unsqueeze(1)
        cx = cam_K[:, 0, 2].unsqueeze(1)
        cy = cam_K[:, 1, 2].unsqueeze(1)

        u = fx * (X / Z_safe) + cx
        v = fy * (Y / Z_safe) + cy
        return u.view(batch_size, Hc, Wc), v.view(batch_size, Hc, Wc), valid.view(batch_size, Hc, Wc)

    def _sample_img_feat(self, img_feat, u_img, v_img, valid, image_hw):
        """
        img_feat: (B,C,Hf,Wf)
        u_img,v_img: pixel coords in resized image space
        image_hw: (H_img, W_img) after resize
        return sampled: (B,C,Hc,Wc), valid_map: (B,1,Hc,Wc)
        """
        
        B, C, Hf, Wf = img_feat.shape
        H_img, W_img = _as_hw(image_hw)

        # map image pixel -> feature pixel
        u_f = u_img / float(self.img_downsample)
        v_f = v_img / float(self.img_downsample)

        # normalize to [-1,1] for grid_sample over feature map
        u_norm = (u_f / max(Wf - 1, 1) ) * 2.0 - 1.0
        v_norm = (v_f / max(Hf - 1, 1) ) * 2.0 - 1.0

        grid = torch.stack([u_norm, v_norm], dim=-1)  # (B,Hc,Wc,2)

        # mask points outside feature map
        inside = (u_f >= 0) & (u_f <= (Wf - 1)) & (v_f >= 0) & (v_f <= (Hf - 1))
        valid2 = valid & inside

        sampled = F.grid_sample(
            img_feat, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B,C,Hc,Wc)

        valid_map = valid2.float().unsqueeze(1)  # (B,1,Hc,Wc)
        sampled = sampled * valid_map
        return sampled, valid_map
    

    def forward(self, img_feat, lidar_bev, calib):
        """
        img_feat: (B,Cimg,Hf,Wf)
        lidar_bev: (B,Clidar,H,W) used ONLY for defining BEV grid size
        calib: dict with keys 'cam_K', 'T_cam_from_lidar' and optionally 'image_hw'
        """
        assert isinstance(calib, dict)
        cam_K = calib.get('cam_K', None)
        T = calib.get('T_cam_from_lidar', None)
        if cam_K is None or T is None:
            raise KeyError("calib must contain 'cam_K' and 'T_cam_from_lidar'")

        device = img_feat.device
        cam_K = cam_K.to(device)
        T = T.to(device)
        delta_xy = calib.get("delta_xy_lidar", None)
        if delta_xy is not None:
            delta_xy = delta_xy.to(device).to(dtype=torch.float32)

        return_valid = bool(calib.get("return_valid", False))




        # resized image H,W (default 1080x1920 like your logs)
        image_hw = calib.get('image_hw', None)
        if image_hw is None:
            # fall back: assume original resized equals (1080,1920)
            image_hw = (1080, 1920)

        xs, ys, Hc, Wc = self._make_bev_centers_xy(lidar_bev, device=device)

        # multi-plane max fusion (if z_list has multiple)
        bev_acc = None
        valid_acc = None
        for z in self.z_list:
            u, v, valid = self._project_points(xs, ys, z, cam_K, T, delta_xy=delta_xy)
            bev_z, vmap = self._sample_img_feat(img_feat, u, v, valid, image_hw=image_hw)
            if bev_acc is None:
                bev_acc = bev_z
                valid_acc = vmap
            else:
                bev_acc = torch.maximum(bev_acc, bev_z)
                valid_acc = torch.maximum(valid_acc, vmap)

            if not hasattr(self, "_dbg_once"):
                self._dbg_once = True
                print(f"[ImageToBEV] valid_mean={float(valid_acc.mean().item()):.4f} z_list={self.z_list}")


        # upsample back to lidar_bev size (STRICT alignment)
        B, Cimg, _, _ = bev_acc.shape
        H, W = lidar_bev.shape[-2], lidar_bev.shape[-1]
        if (Hc != H) or (Wc != W):
            bev_acc = F.interpolate(bev_acc, size=(H, W), mode='bilinear', align_corners=False)
            valid_acc = F.interpolate(valid_acc, size=(H, W), mode='nearest')

        if self.debug_save:
            os.makedirs('tools/output', exist_ok=True)
            self._save_debug_heatmap(bev_acc, 'tools/output/img_bev_heatmap.png')
            # roi mask (valid map) just for sanity
            vm = (valid_acc[0, 0].detach().float().clamp(0, 1) * 255.0).byte().cpu().numpy()
            import cv2
            cv2.imwrite('tools/output/valid_mask.png', vm)
        
        if calib.get('return_valid', False):
            return bev_acc, valid_acc


        return bev_acc
