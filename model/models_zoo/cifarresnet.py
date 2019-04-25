"""ResNets, implemented in PyTorch."""
from __future__ import division

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _conv3x3, _bn_no_affine

__all__ = ['get_cifar_resnet',
           'cifar_resnet20_v1', 'cifar_resnet56_v1', 'cifar_resnet110_v1',
           'cifar_resnet20_v2', 'cifar_resnet56_v2', 'cifar_resnet110_v2']


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
class CIFARBasicBlockV1(nn.Module):
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
        Whether to down sample the input.
    norm_layer : object
        Normalization layer used (default: nn.BatchNorm2d)
        Can be :class:`nn.BatchNorm` or :class:`other normalization layer`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False, **kwargs):
        super(CIFARBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.Sequential(_conv3x3(in_channels, channels, stride),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                  _conv3x3(channels, channels, 1),
                                  nn.BatchNorm2d(channels))
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(channels))
        else:
            self.downsample = None

    def forward(self, x):
        """forward"""
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.relu(residual + x)

        return x


class CIFARBasicBlockV2(nn.Module):
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
        Whether to down sample the input.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, in_channels, channels, stride, downsample=False, **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels, 1)
        if downsample:
            self.downsample = nn.Conv2d(in_channels, channels, 1, stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        """forward"""
        residual = x

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)
        return x + residual


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class CIFARResNetV1(nn.Module):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : nn.Module
        Class for the residual block. Options are CIFARBasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be two larger than layers list.
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, block, layers, channels, classes=10, **kwargs):
        super(CIFARResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 2
        self.features = list()
        self.features.append(nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=False))
        self.features.append(nn.BatchNorm2d(channels[1]))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(block, num_layer, channels[i + 1], channels[i + 2], stride))

        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, in_channels, channels, stride):
        layer = list()
        layer.append(block(in_channels, channels, stride, channels != in_channels))
        for _ in range(layers - 1):
            layer.append(block(channels, channels, 1, False))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze_(3).squeeze_(2)
        x = self.output(x)

        return x


class CIFARResNetV2(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : nn.Module
        Class for the residual block. Options are CIFARBasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be two larger than layers list.
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, block, layers, channels, classes=10, **kwargs):
        super(CIFARResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 2
        self.features = list()
        self.features.append(_bn_no_affine(channels[0]))
        self.features.append(nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=False))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(block, num_layer, channels[i + 1], channels[i + 2], stride))
        self.features.append(nn.BatchNorm2d(channels[-1]))
        self.features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, in_channels, channels, stride):
        layer = list()
        layer.append(block(in_channels, channels, stride, channels != in_channels))
        for _ in range(layers - 1):
            layer.append(block(channels, channels, 1, False))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze_(3).squeeze_(2)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
resnet_net_versions = [CIFARResNetV1, CIFARResNetV2]
resnet_block_versions = [CIFARBasicBlockV1, CIFARBasicBlockV2]


def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [3, 16, 16, 32, 64]
    layers = [n] * (len(channels) - 2)
    return layers, channels


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_cifar_resnet(version, num_layers, pretrained=False,
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
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    layers, channels = _get_resnet_spec(num_layers)

    resnet_class = resnet_net_versions[version - 1]
    block_class = resnet_block_versions[version - 1]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('cifar_resnet%d_v%d' % (num_layers, version), root=root),
                                       map_location=lambda storage, loc: storage))
    return net


def cifar_resnet20_v1(**kwargs):
    r"""ResNet-20 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(1, 20, **kwargs)


def cifar_resnet56_v1(**kwargs):
    r"""ResNet-56 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(1, 56, **kwargs)


def cifar_resnet110_v1(**kwargs):
    r"""ResNet-110 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(1, 110, **kwargs)


def cifar_resnet20_v2(**kwargs):
    r"""ResNet-20 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(2, 20, **kwargs)


def cifar_resnet56_v2(**kwargs):
    r"""ResNet-56 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(2, 56, **kwargs)


def cifar_resnet110_v2(**kwargs):
    r"""ResNet-110 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`other normalizations`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """
    return get_cifar_resnet(2, 110, **kwargs)

# if __name__ == '__main__':
#     import torch
#
#     a = torch.randn(2, 3, 40, 40)
#
#     net1 = cifar_resnet20_v1()
#     net2 = cifar_resnet20_v2()
#     net3 = cifar_resnet56_v1()
#     net4 = cifar_resnet56_v2()
#     net5 = cifar_resnet110_v1()
#     net6 = cifar_resnet110_v2()
#     net1(a)
#     net2(a)
#     net3(a)
#     net4(a)
#     net5(a)
#     net6(a)
