import torch
from torch import nn
import torch.nn.functional as F

from model.module.basic import Residual, _layer, _layer_reverse
from model.module.basic import _pool_layer, _unpool_layer, _merge_layer


# module function
class BasicConv(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(BasicConv, self).__init__()
        p = (k - 1) // 2
        self.with_bn = with_bn
        self.conv = nn.Conv2d(inp_dim, out_dim, k, stride, p, bias=not with_bn)
        if with_bn:
            self.bn = norm_layer(out_dim, **({} if norm_kwargs is None else norm_kwargs))

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return F.relu(x)


class FireModule(nn.Module):
    def __init__(self, inp_dim, out_dim, sr=2, stride=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(FireModule, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_dim // sr, **({} if norm_kwargs is None else norm_kwargs))
        self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1,
                                  stride=stride, groups=out_dim // sr, bias=False)
        self.bn2 = norm_layer(out_dim, **({} if norm_kwargs is None else norm_kwargs))
        self.skip = (stride == 1 and inp_dim == out_dim)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = torch.cat((self.conv_1x1(out), self.conv_3x3(out)), 1)
        out = self.bn2(out)
        if self.skip:
            return F.relu(out + x)
        else:
            return F.relu(out)


# basic hourglass module
class HGModule(nn.Module):
    def __init__(self, num, channels, mod_nums,
                 up_layer=_layer, low_layer=_layer, merge_layer=_merge_layer,
                 pool_layer=_pool_layer, unpool_layer=_unpool_layer,
                 hg_layer=_layer, hg_layer_reverse=_layer_reverse,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(HGModule, self).__init__()
        mod_cur, mod_next = mod_nums[0], mod_nums[1]
        c_cur, c_next = channels[0], channels[1]

        self.num = num
        self.up1 = up_layer(c_cur, c_cur, mod_cur)
        self.max1 = pool_layer(c_cur)
        self.low1 = hg_layer(c_cur, c_next, mod_cur, norm_layer, norm_kwargs)
        if num > 1:
            self.low2 = HGModule(num - 1, channels[1:], low_layer, merge_layer, pool_layer,
                                 unpool_layer, hg_layer, hg_layer_reverse, norm_layer, norm_kwargs)
        else:
            self.low2 = low_layer(c_next, c_next, mod_next, norm_layer, norm_kwargs)
        self.low3 = hg_layer_reverse(c_next, c_cur, mod_cur, norm_layer, norm_kwargs)
        self.up2 = unpool_layer(c_cur)
        self.merge = merge_layer(c_cur)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)


class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()
        self.pre = nn.Sequential(BasicConv(7, 3, 128, 2), Residual(128, 256, stride=2),
                                 Residual(256, 256, stride=2))
