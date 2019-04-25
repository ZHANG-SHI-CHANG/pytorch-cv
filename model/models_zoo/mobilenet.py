"""MobileNet and MobileNetV2, implemented in PyTorch."""

__all__ = ['MobileNet', 'MobileNetV2',
           'mobilenet1_0', 'mobilenet_v2_1_0',
           'mobilenet0_75', 'mobilenet_v2_0_75',
           'mobilenet0_5', 'mobilenet_v2_0_5',
           'mobilenet0_25', 'mobilenet_v2_0_25',
           'get_mobilenet', 'get_mobilenet_v2']

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _add_conv, _add_conv_dw


# -----------------------------------------------------------------------------
# BOTTLENECK
# -----------------------------------------------------------------------------
class LinearBottleneck(nn.Module):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.out = list()
        _add_conv(self.out, in_channels, in_channels * t, relu6=True)
        _add_conv(self.out, in_channels * t, in_channels * t, kernel=3,
                  stride=stride, pad=1, num_group=in_channels * t, relu6=True)
        _add_conv(self.out, in_channels * t, channels, active=False, relu6=True)
        self.out = nn.Sequential(*self.out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = out + x
        return out


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class MobileNet(nn.Module):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        self.features = list()
        _add_conv(self.features, 3, channels=int(32 * multiplier), kernel=3, pad=1, stride=2)
        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2 +
                       [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in [64] + [128] * 2 +
                    [256] * 2 + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        in_channels = int(32 * multiplier)
        for dwc, c, s in zip(dw_channels, channels, strides):
            _add_conv_dw(self.features, in_channels, dw_channels=dwc, channels=c, stride=s)
            in_channels = c
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(channels[-1], classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(3).squeeze(2)
        x = self.output(x)
        return x


class MobileNetV2(nn.Module):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        self.features = list()
        _add_conv(self.features, 3, int(32 * multiplier), kernel=3, stride=2,
                  pad=1, relu6=True)

        in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                             + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                          + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
        ts = [1] + [6] * 16
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

        for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
            self.features.append(LinearBottleneck(in_channels=in_c, channels=c, t=t, stride=s))

        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        _add_conv(self.features, channels_group[-1], last_channels, relu6=True)

        self.features = nn.Sequential(*self.features)

        self.output = nn.Conv2d(last_channels, classes, 1, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.output(x).squeeze(3).squeeze(2)
        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_mobilenet(multiplier, pretrained=False, root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default ~/.torch/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    net = MobileNet(multiplier, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_state_dict(torch.load(get_model_file('mobilenet%s' % version_suffix, root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def get_mobilenet_v2(multiplier, pretrained=False, root=os.path.expanduser('~/.torch/models'),
                     **kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default ~/.torch/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    net = MobileNetV2(multiplier, **kwargs)

    if pretrained:
        import torch
        from model.model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_state_dict(torch.load(get_model_file('mobilenetv2_%s' % version_suffix, root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def mobilenet1_0(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 1.0.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet(1.0, **kwargs)


def mobilenet_v2_1_0(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet_v2(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.75.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet(0.75, **kwargs)


def mobilenet_v2_0_75(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet_v2(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.5.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet(0.5, **kwargs)


def mobilenet_v2_0_5(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet_v2(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.25.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet(0.25, **kwargs)


def mobilenet_v2_0_25(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    return get_mobilenet_v2(0.25, **kwargs)


if __name__ == '__main__':
    net = mobilenet1_0()
    # print(net)
    # print(len(list(net.features.children())))
    net1 = mobilenet0_25()
    net2 = mobilenet0_5()
    net3 = mobilenet0_75()
    net4 = mobilenet1_0()
    net5 = mobilenet_v2_0_5()
    net6 = mobilenet_v2_0_25()
    net7 = mobilenet_v2_0_75()
    net8 = mobilenet_v2_1_0()
    # # print(net8)
    import torch

    a = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        net1(a)
        net2(a)
        net3(a)
        net4(a)
        net5(a)
        net6(a)
        net7(a)
        net8(a)
