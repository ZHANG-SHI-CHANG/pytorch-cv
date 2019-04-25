"""SENet, implemented in PyTorch."""
from __future__ import division

__all__ = ['SENet', 'SEBlock', 'get_senet', 'senet_154']

import os
import math
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    r"""SEBlock from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, in_channels, channels, cardinality, bottleneck_width, stride,
                 downsample=False, downsample_kernel_size=3, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = list()
        self.body.append(nn.Conv2d(in_channels, group_width // 2, kernel_size=1, bias=False))
        self.body.append(nn.BatchNorm2d(group_width // 2))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(group_width // 2, group_width, kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False))
        self.body.append(nn.BatchNorm2d(group_width))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(group_width, channels * 4, kernel_size=1, bias=False))
        self.body.append(nn.BatchNorm2d(channels * 4))
        self.body = nn.Sequential(*self.body)

        self.se = list()
        self.se.append(nn.Conv2d(channels * 4, channels // 4, kernel_size=1, padding=0))
        self.se.append(nn.ReLU(inplace=True))
        self.se.append(nn.Conv2d(channels // 4, channels * 4, kernel_size=1, padding=0))
        self.se.append(nn.Sigmoid())
        self.se = nn.Sequential(*self.se)

        if downsample:
            self.downsample = list()
            downsample_padding = 1 if downsample_kernel_size == 3 else 0
            self.downsample.append(nn.Conv2d(in_channels, channels * 4, kernel_size=downsample_kernel_size,
                                             stride=stride, padding=downsample_padding, bias=False))
            self.downsample.append(nn.BatchNorm2d(channels * 4))
            self.downsample = nn.Sequential(*self.downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.body(x)

        w = F.adaptive_avg_pool2d(x, 1)
        w = self.se(w)
        x = x * w

        if self.downsample:
            residual = self.downsample(residual)

        x = F.relu(x + residual)
        return x


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class SENet(nn.Module):
    r"""ResNext model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 1000
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments..
    """

    def __init__(self, layers, cardinality, bottleneck_width,
                 classes=1000, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels, in_channels = 64, 128

        self.features = list()
        self.features.append(nn.Conv2d(3, channels, 3, 2, 1, bias=False))
        self.features.append(nn.BatchNorm2d(channels))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
        self.features.append(nn.BatchNorm2d(channels))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False))
        self.features.append(nn.BatchNorm2d(channels * 2))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(3, 2, ceil_mode=True))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(in_channels, channels, num_layer, stride, i + 1))
            channels, in_channels = channels * 2, in_channels * 2
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels * 2, classes)

    def _make_layer(self, in_channels, channels, num_layers, stride, stage_index):
        layer = list()
        downsample_kernel_size = 1 if stage_index == 1 else 3
        layer.append(SEBlock(in_channels, channels, self.cardinality, self.bottleneck_width,
                             stride, True, downsample_kernel_size))
        for _ in range(num_layers - 1):
            layer.append(SEBlock(channels * 4, channels, self.cardinality, self.bottleneck_width,
                                 1, False))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(3).squeeze(2)
        x = F.dropout(x, 0.2)
        x = self.output(x)

        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3],
                152: [3, 8, 36, 3]}


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_senet(num_layers, cardinality=64, bottleneck_width=4, pretrained=False,
              root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers.
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    assert num_layers in resnext_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnext_spec.keys()))
    layers = resnext_spec[num_layers]
    net = SENet(layers, cardinality, bottleneck_width, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('senet_%d' % (num_layers + 2), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def senet_154(**kwargs):
    r"""SENet 154 model from
    `"Squeeze-and-excitation networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_senet(152, **kwargs)


if __name__ == '__main__':
    net = senet_154()
    import torch

    # print(net)

    a = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(a)
    print(out.shape)
