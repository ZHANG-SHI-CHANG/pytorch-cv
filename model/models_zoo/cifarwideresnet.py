"""ResNets, implemented in PyTorch."""
from __future__ import division

__all__ = ['get_cifar_wide_resnet', 'cifar_wideresnet16_10',
           'cifar_wideresnet28_10', 'cifar_wideresnet40_8']

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _conv3x3, _bn_no_affine


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
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
        Whether to downsample the input.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, in_channels, channels, stride, downsample=False, drop_rate=0.0,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn2 = norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, channels, 1)
        self.droprate = drop_rate
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
        if self.droprate > 0:
            x = F.dropout(x, self.droprate)
        x = self.conv2(x)

        return x + residual


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class CIFARWideResNet(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : nn.Module
        Class for the residual block. Options are CIFARBasicBlockV2
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, block, layers, channels, drop_rate, classes=10,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(CIFARWideResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 2
        self.features = list()
        self.features.append(_bn_no_affine(channels[0], **({} if norm_kwargs is None else norm_kwargs)))
        self.features.append(nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=False))
        self.features.append(norm_layer(channels[1], **({} if norm_kwargs is None else norm_kwargs)))

        in_channels = channels[1]
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(self._make_layer(block, num_layer, in_channels, channels[i + 2], drop_rate,
                                                  stride, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_channels = channels[i + 2]
        self.features.append(norm_layer(channels[-1], **({} if norm_kwargs is None else norm_kwargs)))
        self.features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, in_channels, channels, drop_rate, stride,
                    norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        layer = list()
        layer.append(block(in_channels, channels, stride, channels != in_channels, drop_rate,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for _ in range(layers - 1):
            layer.append(block(channels, channels, 1, False, drop_rate,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze(3).squeeze(2)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_cifar_wide_resnet(num_layers, width_factor=1, drop_rate=0.0, pretrained=False,
                          root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    width_factor: int
        The width factor to apply to the number of channels from the original resnet.
    drop_rate: float
        The rate of dropout.
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
    assert (num_layers - 4) % 6 == 0

    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [3, 16, 16 * width_factor, 32 * width_factor, 64 * width_factor]

    net = CIFARWideResNet(CIFARBasicBlockV2, layers, channels, drop_rate, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('cifar_wideresnet%d_%d' % (num_layers, width_factor),
                                                      root=root), map_location=lambda storage, loc: storage))
    return net


def cifar_wideresnet16_10(**kwargs):
    r"""WideResNet-16-10 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
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
    return get_cifar_wide_resnet(16, 10, **kwargs)


def cifar_wideresnet28_10(**kwargs):
    r"""WideResNet-28-10 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
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
    return get_cifar_wide_resnet(28, 10, **kwargs)


def cifar_wideresnet40_8(**kwargs):
    r"""WideResNet-40-8 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
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
    return get_cifar_wide_resnet(40, 8, **kwargs)


if __name__ == '__main__':
    import torch

    a = torch.randn(2, 3, 40, 40)

    net1 = cifar_wideresnet16_10()
    net2 = cifar_wideresnet28_10()
    net3 = cifar_wideresnet40_8()

    with torch.no_grad():
        net1(a)
        net2(a)
        net3(a)
