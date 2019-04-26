import torch
from torch import nn
import torch.nn.functional as F

from model.module.basic import Residual, BasicConv, _layer, _layer_reverse
from model.module.basic import _pool_layer, _unpool_layer, _merge_layer


class FireModule(nn.Module):
    def __init__(self, inp_dim, out_dim, sr=2, stride=1):
        super(FireModule, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1,
                                  stride=stride, groups=out_dim // sr, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.skip = (stride == 1 and inp_dim == out_dim)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = torch.cat((self.conv_1x1(out), self.conv_3x3(out)), 1)
        out = self.bn2(out)
        if self.skip:
            return F.relu(out + x)
        else:
            return F.relu(out)


def layer(inp_dim, out_dim, num):
    layers = [FireModule(inp_dim, out_dim)]
    layers += [FireModule(out_dim, out_dim) for _ in range(1, num)]
    return nn.Sequential(*layers)


def layer_reverse(inp_dim, out_dim, num):
    layers = [FireModule(inp_dim, inp_dim) for _ in range(num - 1)]
    layers += [FireModule(inp_dim, out_dim)]
    return nn.Sequential(*layers)


def pool_layer(dim):
    return nn.Sequential()


def unpool_layer(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def hg_layer(inp_dim, out_dim, num):
    layers = [FireModule(inp_dim, out_dim, stride=2)]
    layers += [FireModule(out_dim, out_dim) for _ in range(1, num)]
    return nn.Sequential(*layers)


# basic hourglass module
class HGModule(nn.Module):
    def __init__(self, num, channels, mod_nums,
                 up_layer=_layer, low_layer=_layer, merge_layer=_merge_layer,
                 pool_layer=_pool_layer, unpool_layer=_unpool_layer,
                 hg_layer=_layer, hg_layer_reverse=_layer_reverse):
        super(HGModule, self).__init__()
        mod_cur, mod_next = mod_nums[0], mod_nums[1]
        c_cur, c_next = channels[0], channels[1]

        self.num = num
        self.up1 = up_layer(c_cur, c_cur, mod_cur)
        self.max1 = pool_layer(c_cur)
        self.low1 = hg_layer(c_cur, c_next, mod_cur)
        if num > 1:
            self.low2 = HGModule(num - 1, channels[1:], mod_nums[1:], up_layer, low_layer, merge_layer,
                                 pool_layer, unpool_layer, hg_layer, hg_layer_reverse)
        else:
            self.low2 = low_layer(c_next, c_next, mod_next)
        self.low3 = hg_layer_reverse(c_next, c_cur, mod_cur)
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
    def __init__(self, stacks=2):
        super(Hourglass, self).__init__()
        self.pre = nn.Sequential(BasicConv(7, 3, 128, 2), Residual(128, 256, stride=2),
                                 Residual(256, 256, stride=2))
        self.hg_modules = nn.ModuleList([
            HGModule(4, [256, 256, 384, 384, 512], [2, 2, 2, 2, 4], up_layer=layer,
                     low_layer=layer, pool_layer=pool_layer, unpool_layer=unpool_layer,
                     hg_layer=hg_layer, hg_layer_reverse=layer_reverse) for _ in range(stacks)])
        self.cnvs = nn.ModuleList([BasicConv(3, 256, 256) for _ in range(stacks)])
        self.inters = nn.ModuleList([Residual(256, 256) for _ in range(stacks - 1)])
        self.cnvs_ = nn.ModuleList([BasicConv(1, 256, 256, with_relu=False) for _ in range(stacks - 1)])
        self.inters_ = nn.ModuleList([BasicConv(1, 256, 256, with_relu=False) for _ in range(stacks - 1)])

    def forward(self, x):
        inter = self.pre(x)

        cnvs = list()
        for ind, (hg_, cnv_) in enumerate(zip(self.hg_modules, self.cnvs)):
            hg = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)

            if ind < len(self.hg_modules) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = F.relu(inter)
                inter = self.inters[ind](inter)
        return cnvs


if __name__ == '__main__':
    net = Hourglass()
    a = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = net(a)
