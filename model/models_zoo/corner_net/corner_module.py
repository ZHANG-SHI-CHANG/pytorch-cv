from torch import nn
import torch.nn.functional as F

from model.module.basic import BasicConv


class CornerPool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(CornerPool, self).__init__()
        self.p1_conv = BasicConv(3, dim, 128)
        self.p2_conv = BasicConv(3, dim, 128)
        self.p_conv = nn.Sequential(
            nn.Conv2d(128, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.conv2 = BasicConv(3, dim, dim)

        self.pool1, self.pool2 = pool1, pool2

    def forward(self, x):
        p1 = self.pool1(self.p1_conv(x))
        p2 = self.pool2(self.p2_conv(x))

        p = self.p_conv(p1 + p2)
        p = F.relu(self.conv1(x) + p)

        return self.conv2(p)


def pred_module(dim):
    return nn.Sequential(
        BasicConv(1, 256, 256, with_bn=False),
        nn.Conv2d(256, dim, 1)
    )
