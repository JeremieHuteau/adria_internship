import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, groups=1, separable=False,
            norm=nn.BatchNorm2d, act=nn.ReLU):
        super(ConvNormAct, self).__init__()

        if separable:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, groups=groups))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups,
                bias=False)

        if norm is None:
            self.norm = None
        else:
            self.norm = norm(out_channels)

        if act is None: 
            self.act = None
        else:
            self.act = act()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

