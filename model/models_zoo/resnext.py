"""ResNext, implemented in PyTorch."""
from __future__ import division

import os
import math
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['ResNext', 'Block', 'get_resnext',
           'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
           'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d']


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
class Block(nn.Module):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    in_channels: int
        input channels
    channels: int
        channels
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """

    def __init__(self, in_channels, channels, cardinality, bottleneck_width, stride,
                 downsample=False, last_gamma=False, use_se=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(Block, self).__init__(**kwargs)
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
        tmp_layer = norm_layer(channels * 4, **({} if norm_kwargs is None else norm_kwargs))
        if last_gamma:
            nn.init.zeros_(tmp_layer.weight)
        self.body.append(tmp_layer)
        self.body = nn.Sequential(*self.body)

        if use_se:
            self.se = list()
            self.se.append(nn.Conv2d(channels * 4, channels // 4, kernel_size=1, padding=0))
            self.se.append(nn.ReLU(inplace=True))
            self.se.append(nn.Conv2d(channels // 4, channels * 4, kernel_size=1, padding=0))
            self.se.append(nn.Sigmoid())
            self.se = nn.Sequential(*self.se)
        else:
            self.se = None

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

        if self.se:
            w = F.adaptive_avg_pool2d(x, output_size=1)
            w = self.se(w)
            x = x * w

        if self.downsample:
            residual = self.downsample(residual)

        x = F.relu(x + residual)
        return x


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class ResNext(nn.Module):
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
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """

    def __init__(self, layers, cardinality, bottleneck_width,
                 classes=1000, last_gamma=False, use_se=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(ResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        in_channels, channels = 64, 64

        self.features = list()
        self.features.append(nn.Conv2d(3, channels, 7, 2, 3, bias=False))

        self.features.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(3, 2, 1))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(in_channels, channels, num_layer, stride, last_gamma,
                                                  use_se, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_channels = in_channels * 4 if i == 0 else in_channels * 2
            channels *= 2
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(in_channels, classes)

    def _make_layer(self, in_channels, channels, num_layers, stride, last_gamma, use_se,
                    norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        layer = list()
        layer.append(Block(in_channels, channels, self.cardinality, self.bottleneck_width,
                           stride, True, last_gamma=last_gamma, use_se=use_se,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for _ in range(num_layers - 1):
            layer.append(Block(channels * 4, channels, self.cardinality, self.bottleneck_width,
                               1, False, last_gamma=last_gamma, use_se=use_se,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze_(3).squeeze_(2)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_resnext(num_layers, cardinality=32, bottleneck_width=4, use_se=False,
                pretrained=False, root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Options are 50, 101.
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
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    assert num_layers in resnext_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnext_spec.keys()))
    layers = resnext_spec[num_layers]
    net = ResNext(layers, cardinality, bottleneck_width, use_se=use_se, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        if not use_se:
            net.load_state_dict(torch.load(get_model_file('resnext%d_%dx%dd' % (num_layers, cardinality,
                                                                                bottleneck_width), root=root)))
        else:
            net.load_state_dict(torch.load(get_model_file('se_resnext%d_%dx%dd' % (num_layers, cardinality,
                                                                                   bottleneck_width), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def resnext50_32x4d(**kwargs):
    r"""ResNext50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

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
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = False
    return get_resnext(50, 32, 4, **kwargs)


def resnext101_32x4d(**kwargs):
    r"""ResNext101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = False
    return get_resnext(101, 32, 4, **kwargs)


def resnext101_64x4d(**kwargs):
    r"""ResNext101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = False
    return get_resnext(101, 64, 4, **kwargs)


def se_resnext50_32x4d(**kwargs):
    r"""SE-ResNext50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = True
    return get_resnext(50, 32, 4, **kwargs)


def se_resnext101_32x4d(**kwargs):
    r"""SE-ResNext101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = True
    return get_resnext(101, 32, 4, **kwargs)


def se_resnext101_64x4d(**kwargs):
    r"""SE-ResNext101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
    """
    kwargs['use_se'] = True
    return get_resnext(101, 64, 4, **kwargs)
