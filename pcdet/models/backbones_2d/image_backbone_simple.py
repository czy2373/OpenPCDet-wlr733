#image_backbone 简单版本
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageBackboneSimple(nn.Module):
    """
    输出一个较低分辨率的特征图，方便后面投影到 BEV。
    """
    def __init__(self, model_cfg):
        super().__init__()
        out_ch = model_cfg.get('OUT_CH', 64)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, batch_dict):
        x = batch_dict['images']   # [B,3,H,W]
        feat = self.net(x)         # [B,C,H/8,W/8]
        batch_dict['image_features'] = feat
        return batch_dict
