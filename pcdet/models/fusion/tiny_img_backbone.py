import torch.nn as nn

class TinyImgBackbone(nn.Module):
    """
    最小可跑相机 backbone：
    输入 (B,3,1080,1920) -> 输出 (B,C,Hi,Wi)
    downsample=4 表示输出分辨率约为 1/4
    """
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
