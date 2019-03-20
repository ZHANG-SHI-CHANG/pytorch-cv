"""ResNets, implemented in PyTorch."""
# TODO: add Squeeze and Excitation module
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'get_resnet']

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _conv3x3, _bn_no_affine


# -----------------------------------------------------------------------------
# BLOCKS & BOTTLENECK
# -----------------------------------------------------------------------------
class BasicBlockV1(nn.Module):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = list()
        self.body.append(_conv3x3(in_channels, channels, stride))
        self.body.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(_conv3x3(channels, channels, 1))
        tmp_layer = norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs))
        if last_gamma:
            nn.init.zeros_(tmp_layer.weight)
        self.body.append(tmp_layer)
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = list()
            self.downsample.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride,
                                             bias=False))
            self.downsample.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
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


class BottleneckV1(nn.Module):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = list()
        self.body.append(nn.Conv2d(in_channels, channels // 4, kernel_size=1, stride=stride))
        self.body.append(norm_layer(channels // 4, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(_conv3x3(channels // 4, channels // 4, 1))
        self.body.append(norm_layer(channels // 4, **({} if norm_kwargs is None else norm_kwargs)))
        self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1))
        tmp_layer = norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs))
        if last_gamma:
            nn.init.zeros_(tmp_layer.weight)
        self.body.append(tmp_layer)
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = list()
            self.downsample.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride,
                                             bias=False, ))
            self.downsample.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
            self.downsample = nn.Sequential(*self.downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.relu(x + residual)
        return x


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn2 = norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs))
        if last_gamma:
            nn.init.zeros_(self.bn2.weight)

        self.conv2 = _conv3x3(channels, channels, 1)

        if downsample:
            self.downsample = nn.Conv2d(in_channels, channels, 1, stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + residual


class BottleneckV2(nn.Module):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = nn.Conv2d(in_channels, channels // 4, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(channels // 4, **({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels // 4, channels // 4, stride)
        self.bn3 = norm_layer(channels // 4, **({} if norm_kwargs is None else norm_kwargs))
        if last_gamma:
            nn.init.zeros_(self.bn3.weight)
        self.conv3 = nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1, bias=False)

        if downsample:
            self.downsample = nn.Conv2d(in_channels, channels, 1, stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)

        return x + residual


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class ResNetV1(nn.Module):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : nn.Module
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be two larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 2

        self.features = list()
        if thumbnail:
            self.features.append(_conv3x3(channels[0], channels[1], 1))
        else:
            self.features.append(nn.Conv2d(channels[0], channels[1], 7, 2, 3, bias=False))
            self.features.append(norm_layer(channels[1], **({} if norm_kwargs is None else norm_kwargs)))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.MaxPool2d(3, 2, 1))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(block, num_layer, channels[i + 1], channels[i + 2],
                                                  stride, last_gamma=last_gamma, norm_layer=norm_layer,
                                                  norm_kwargs=norm_kwargs))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, in_channels, channels, stride,
                    last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        layer = list()

        layer.append(block(in_channels, channels, stride, channels != in_channels,
                           last_gamma=last_gamma, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for _ in range(layers - 1):
            layer.append(block(channels, channels, 1, False, last_gamma=last_gamma,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze_(3).squeeze_(2)
        x = self.output(x)
        return x


class ResNetV2(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : nn.Module
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be two larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 2

        self.features = list()
        self.features.append(_bn_no_affine(channels[0], norm_layer, norm_kwargs))
        if thumbnail:
            self.features.append(_conv3x3(channels[0], channels[1], 1))
        else:
            self.features.append(nn.Conv2d(channels[0], channels[1], 7, 2, 3, bias=False))
            self.features.append(norm_layer(channels[1], **({} if norm_kwargs is None else norm_kwargs)))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.MaxPool2d(3, 2, 1))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(block, num_layer, channels[i + 1], channels[i + 2],
                                                  stride, last_gamma=last_gamma, norm_layer=norm_layer,
                                                  norm_kwargs=norm_kwargs))
        self.features.append(norm_layer(channels[-1], **({} if norm_kwargs is None else norm_kwargs)))
        self.features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, in_channels, channels, stride,
                    last_gamma=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        layer = list()
        layer.append(block(in_channels, channels, stride, channels != in_channels,
                           last_gamma=last_gamma, norm_layer=norm_layer,
                           norm_kwargs=norm_kwargs))
        for _ in range(layers - 1):
            layer.append(block(channels, channels, 1, False, last_gamma=last_gamma,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze_(3).squeeze_(2)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [3, 64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [3, 64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [3, 64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [3, 64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [3, 64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_resnet(version, num_layers, pretrained=False,
               root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default $~/.torch/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2." % version
    resnet_class = resnet_net_versions[version - 1]
    block_class = resnet_block_versions[version - 1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('resnet%d_v%d' % (num_layers, version),
                                                      root=root), map_location=lambda storage, loc: storage))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def resnet18_v1(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(1, 18, **kwargs)


def resnet34_v1(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(1, 34, **kwargs)


def resnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(1, 50, **kwargs)


def resnet101_v1(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(1, 101, **kwargs)


def resnet152_v1(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(1, 152, **kwargs)


def resnet18_v2(**kwargs):
    r"""ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(2, 18, **kwargs)


def resnet34_v2(**kwargs):
    r"""ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(2, 34, **kwargs)


def resnet50_v2(**kwargs):
    r"""ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(2, 50, **kwargs)


def resnet101_v2(**kwargs):
    r"""ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(2, 101, **kwargs)


def resnet152_v2(**kwargs):
    r"""ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_resnet(2, 152, **kwargs)


if __name__ == '__main__':
    import torch

    net = resnet50_v1()
    # cnt = 0
    # for key in net.state_dict().keys():
    #     if not key.endswith('num_batches_tracked'):
    #         print(key)

    # a = torch.randn(2, 3, 224, 224)

    # net1 = resnet18_v1()
    # net2 = resnet18_v2()
    # net3 = resnet34_v1()
    # net4 = resnet34_v2()
    # net5 = resnet50_v1()
    # net6 = resnet50_v2()
    # net7 = resnet101_v1()
    # net8 = resnet101_v2()
    # net9 = resnet152_v1()
    # net10 = resnet152_v2()
    # with torch.no_grad():
    #     net1(a)
    #     net2(a)
    #     net3(a)
    #     net4(a)
    #     net5(a)
    #     net6(a)
    #     net7(a)
    #     net8(a)
    #     net9(a)
    #     net10(a)
