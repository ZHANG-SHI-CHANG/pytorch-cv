"""ResNets, implemented in PyTorch."""
from __future__ import division

__all__ = ['get_cifar_resnext', 'cifar_resnext29_32x4d', 'cifar_resnext29_16x64d']

import os
import math

from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
class CIFARBlock(nn.Module):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
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
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, cardinality, bottleneck_width,
                 stride, downsample=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(CIFARBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = list()
        self.body.append(nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False))
        self.body.append(norm_layer(group_width, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False))
        self.body.append(norm_layer(group_width, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(group_width, channels * 4, kernel_size=1, bias=False))
        self.body.append(norm_layer(channels * 4, **({} if norm_kwargs is None else norm_kwargs)))
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = list()
            self.downsample.append(nn.Conv2d(in_channels, channels * 4, kernel_size=1, stride=stride,
                                             bias=False))
            self.downsample.append(norm_layer(channels * 4, **({} if norm_kwargs is None else norm_kwargs)))
            self.downsample = nn.Sequential(*self.downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.relu(residual + x)

        return x


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class CIFARResNext(nn.Module):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, layers, cardinality, bottleneck_width, classes=10,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(CIFARResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        in_channels, channels = 64, 64

        self.features = list()
        self.features.append(nn.Conv2d(3, channels, 3, 1, 1, bias=False))
        self.features.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.features.append(nn.ReLU(inplace=True))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(in_channels, channels, num_layer, stride,
                                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_channels = in_channels * 4 if i == 0 else in_channels * 2
            channels *= 2
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(in_channels, classes)

    def _make_layer(self, in_channels, channels, num_layer, stride,
                    norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        layer = list()
        layer.append(CIFARBlock(in_channels, channels, self.cardinality, self.bottleneck_width,
                                stride, True, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        in_channels = channels * 4
        for _ in range(num_layer - 1):
            layer.append(CIFARBlock(in_channels, channels, self.cardinality, self.bottleneck_width,
                                    1, False, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze(3).squeeze(2)
        x = self.output(x)

        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_cifar_resnext(num_layers, cardinality=16, bottleneck_width=64, pretrained=False,
                      root=os.path.join(os.path.expanduser('~'), '.torch', 'models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    assert (num_layers - 2) % 9 == 0
    layer = (num_layers - 2) // 9
    layers = [layer] * 3
    net = CIFARResNext(layers, cardinality, bottleneck_width, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('cifar_resnext%d_%dx%dd' %
                                                      (num_layers, cardinality, bottleneck_width), root=root)))
    return net


def cifar_resnext29_32x4d(**kwargs):
    r"""ResNext-29 32x4d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_cifar_resnext(29, 32, 4, **kwargs)


def cifar_resnext29_16x64d(**kwargs):
    r"""ResNext-29 16x64d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_cifar_resnext(29, 16, 64, **kwargs)


if __name__ == '__main__':
    import torch

    a = torch.randn(2, 3, 40, 40)
    net1 = cifar_resnext29_16x64d()
    # print(net1)
    net2 = cifar_resnext29_32x4d()
    with torch.no_grad():
        net1(a)
        net2(a)
