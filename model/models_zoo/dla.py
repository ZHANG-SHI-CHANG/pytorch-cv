import os

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['get_dla', 'dla34']


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(BasicBlock, self).__init__()
        self.body = list()
        self.body.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=dilation, bias=False, dilation=dilation))
        self.body.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=dilation, bias=False, dilation=dilation))
        self.body.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
        self.body = nn.Sequential(*self.body)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        x = self.body(x)
        x = F.relu(x + residual)

        return x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        body = list()
        body.append(nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False))
        body.append(norm_layer(bottle_planes, **({} if norm_kwargs is None else norm_kwargs)))
        body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                              stride=stride, padding=dilation,
                              bias=False, dilation=dilation))
        body.append(norm_layer(bottle_planes, **({} if norm_kwargs is None else norm_kwargs)))
        body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(bottle_planes, planes,
                              kernel_size=1, bias=False))
        body.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
        self.body = nn.Sequential(*body)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        x = self.body(x)
        x = F.relu(x + residual)

        return x


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        body = list()
        body.append(nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False))
        body.append(norm_layer(bottle_planes, **({} if norm_kwargs is None else norm_kwargs)))
        body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                              stride=stride, padding=dilation, bias=False,
                              dilation=dilation, groups=cardinality))
        body.append(norm_layer(bottle_planes, **({} if norm_kwargs is None else norm_kwargs)))
        body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False))
        body.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
        self.body = nn.Sequential(*body)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        x = self.body(x)

        x = F.relu(x + residual)

        return x


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1,
                              stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False,
                 root_dim=0, root_kernel_size=1, dilation=1, root_residual=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0, root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(channels[0], **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True))
        level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                      level_root=False, root_residual=residual_root,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                      level_root=True, root_residual=residual_root,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                      level_root=True, root_residual=residual_root,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                      level_root=True, root_residual=residual_root,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.features = nn.Sequential(base_layer, level0, level1, level2, level3, level3,
                                      level4, level5)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                self.norm_layer(planes, **({} if self.norm_kwargs is None else self.norm_kwargs)),
            )

        layers = list()
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                self.norm_layer(planes, **({} if self.norm_kwargs is None else self.norm_kwargs)),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc(x).view(x.shape[0], -1)
        return x


def get_dla(name, levels, channels, block, pretrained=False,
            root=os.path.expanduser('~/.torch/models'), **kwargs):
    net = DLA(levels, channels, block=block, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(name, root=root)))
    return net


def dla34(**kwargs):
    return get_dla('dla34', [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                   block=BasicBlock, **kwargs)


if __name__ == '__main__':
    net = dla34()
    print(net)
    # a = torch.randn(1, 3, 224, 224)
    # print(net(a).shape)
