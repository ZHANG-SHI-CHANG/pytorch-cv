"""Darknet as YOLO backbone network."""
from __future__ import absolute_import

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _conv2d

__all__ = ['DarknetV3', 'get_darknet', 'darknet53']


# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------
class DarknetBasicBlockV3(nn.Module):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv.

    Parameters
    ----------
    in_channel : int
        input channels.
    channel : int
        Convolution channels for 1x1 conv.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.

    """

    def __init__(self, in_channel, channel, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = list()
        # 1x1 reduce
        self.body.append(_conv2d(in_channel, channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        # 3x3 conv expand
        self.body.append(_conv2d(channel, channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        residual = x
        x = self.body(x)
        return x + residual


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class DarknetV3(nn.Module):
    """Darknet v3.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    classes : int, default is 1000
        Number of classes, which determines the dense layer output channels.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.

    Attributes
    ----------
    features : nn.Module
        Feature extraction layers.
    output : nn.Linear
        A classes(1000)-way Fully-Connected Layer.

    """

    def __init__(self, layers, channels, classes=1000,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DarknetV3, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        self.features = list()
        # first 3x3 conv
        self.features.append(_conv2d(3, channels[0], 3, 1, 1,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for i, (nlayer, channel) in enumerate(zip(layers, channels[1:])):
            assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
            # add downsample conv with stride=2
            self.features.append(_conv2d(channels[i], channel, 3, 1, 2,
                                         norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # add nlayer basic blocks
            for i in range(nlayer):
                self.features.append(DarknetBasicBlockV3(channel, channel // 2,
                                                         norm_layer=norm_layer, norm_kwargs=None))
        self.features = nn.Sequential(*self.features)
        # output
        self.output = nn.Linear(channels[-1], classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2]).squeeze(3).squeeze(2)
        return self.output(x)


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
darknet_versions = {'v3': DarknetV3}
darknet_spec = {
    'v3': {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024]), }
}


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_darknet(darknet_version, num_layers, pretrained=False,
                root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Get darknet by `version` and `num_layers` info.

    Parameters
    ----------
    darknet_version : str
        Darknet version, choices are ['v3'].
    num_layers : int
        Number of layers.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.

    Returns
    -------
    nn.Module
        Darknet network.

    Examples
    --------
    >>> model = get_darknet('v3', 53, pretrained=True)
    >>> print(model)

    """
    assert darknet_version in darknet_versions and darknet_version in darknet_spec, (
        "Invalid darknet version: {}. Options are {}".format(
            darknet_version, str(darknet_versions.keys())))
    specs = darknet_spec[darknet_version]
    assert num_layers in specs, (
        "Invalid number of layers: {}. Options are {}".format(num_layers, str(specs.keys())))
    layers, channels = specs[num_layers]
    darknet_class = darknet_versions[darknet_version]
    net = darknet_class(layers, channels, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        from data.imagenet import ImageNetAttr
        net.load_state_dict(torch.load(get_model_file('darknet%d' % num_layers, root=root)))
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def darknet53(**kwargs):
    """Darknet v3 53 layer network.
    Reference: https://arxiv.org/pdf/1804.02767.pdf.

    Parameters
    ----------
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments

    Returns
    -------
    nn.Module
        Darknet network.

    """
    return get_darknet('v3', 53, **kwargs)


if __name__ == '__main__':
    net = darknet53()
    print(net)
