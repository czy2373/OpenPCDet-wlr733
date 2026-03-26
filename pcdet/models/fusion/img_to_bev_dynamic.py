# 会根据 spatial_features_2d 的尺寸自动计算 BEV cell 间距（等价于 stride）
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgToBEVDynamic(nn.Module):
    """
    把 image feature 通过投影采样成与 target_bev_feat 同尺寸的 BEV feature。
    - 使用固定高度 z0（最小可跑版本）
    - calib 里用 K 和 T_cam_from_lidar（cam = T * lidar）
    """
    def __init__(self, pc_range, voxel_size, z0=0.0, img_downsample=4):
        super().__init__()
        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = [float(x) for x in pc_range]
        self.vx, self.vy = float(voxel_size[0]), float(voxel_size[1])
        self.z0 = float(z0)
        self.img_downsample = int(img_downsample)

        # 预存原始网格尺寸（由 range/voxel 决定）
        self.nx0 = int(round((self.xmax - self.xmin) / self.vx))  # 1056
        self.ny0 = int(round((self.ymax - self.ymin) / self.vy))  # 1320

    @torch.no_grad()
    def _make_bev_points(self, B, H, W, device, dtype):
        """
        根据 target (H,W) 推导 stride（相对原始 Scatter 网格），并生成 (B,H,W,4) LiDAR 坐标点
        """
        # stride：原始网格到目标网格的缩放比
        sx = self.nx0 / float(W)
        sy = self.ny0 / float(H)

        # 每个目标 cell 对应的米尺度
        vx_eff = self.vx * sx
        vy_eff = self.vy * sy

        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * vx_eff + self.xmin
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * vy_eff + self.ymin
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)

        ones = torch.ones_like(xx)
        zz = torch.full_like(xx, self.z0)

        pts = torch.stack([xx, yy, zz, ones], dim=-1)     # (H,W,4)
        pts = pts.unsqueeze(0).repeat(B, 1, 1, 1)         # (B,H,W,4)
        return pts

    def forward(self, img_feat, target_bev_feat, calib):
        """
        img_feat: (B, Ci, Hi, Wi)
        target_bev_feat: (B, Cl, H, W)  用它的 H,W 来对齐
        calib: {'K': (B,3,3), 'T_cam_from_lidar': (B,4,4)}
        return: img_bev (B, Ci, H, W)
        """
        B, Ci, Hi, Wi = img_feat.shape
        _, _, H, W = target_bev_feat.shape

        device = img_feat.device
        dtype = img_feat.dtype

        K = calib['K'].to(device=device, dtype=dtype)                     # (B,3,3)
        T = calib['T_cam_from_lidar'].to(device=device, dtype=dtype)      # (B,4,4)

        pts = self._make_bev_points(B, H, W, device, dtype)               # (B,H,W,4)
        pts = pts.view(B, -1, 4).transpose(1, 2)                          # (B,4,HW)

        cam = torch.matmul(T, pts)                                         # (B,4,HW)
        X = cam[:, 0, :]
        Y = cam[:, 1, :]
        Z = cam[:, 2, :].clamp(min=1e-3)

        fx = K[:, 0, 0].unsqueeze(-1)
        fy = K[:, 1, 1].unsqueeze(-1)
        cx = K[:, 0, 2].unsqueeze(-1)
        cy = K[:, 1, 2].unsqueeze(-1)

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        # 映射到 feature map 坐标（img_feat 是原图下采样 img_downsample 得到）
        uf = u / self.img_downsample
        vf = v / self.img_downsample

        x_norm = (uf / (Wi - 1)) * 2 - 1
        y_norm = (vf / (Hi - 1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm], dim=-1)                      # (B,HW,2)
        grid = grid.view(B, H, W, 2)

        img_bev = F.grid_sample(img_feat, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        return img_bev
