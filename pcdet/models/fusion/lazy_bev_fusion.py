# 这个模块会在第一次 forward 时根据 lidar_bev 的通道数自动建 1x1 conv（
import torch
import torch.nn as nn

class LazyBEVFusion(nn.Module):
    def __init__(self, img_channels, out_channels=None):
        super().__init__()
        self.img_channels = int(img_channels)
        self.out_channels = out_channels  # None 表示输出通道=lidar通道
        self.fuse = None  # lazy build

    def _build(self, lidar_channels, device):
        out_c = lidar_channels if self.out_channels is None else int(self.out_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(lidar_channels + self.img_channels, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ).to(device)

    def forward(self, lidar_bev, img_bev):
        if self.fuse is None:
            self._build(lidar_bev.shape[1], lidar_bev.device)
        x = torch.cat([lidar_bev, img_bev], dim=1)
        return self.fuse(x)
