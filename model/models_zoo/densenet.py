"""DenseNet, implemented in PyTorch."""

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import _make_dense_block, _make_transition


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class DenseNet(nn.Module):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments
    """

    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1000, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.features = list()
        self.features.append(nn.Conv2d(3, num_init_features, kernel_size=7,
                                       stride=2, padding=3, bias=False))
        self.features.append(nn.BatchNorm2d(num_init_features))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # Add dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.append(_make_dense_block(num_features, num_layers, bn_size, growth_rate, dropout))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.append(_make_transition(num_features, num_features // 2))
                num_features = num_features // 2
        self.features.append(nn.BatchNorm2d(num_features))
        self.features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(num_features, classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(3).squeeze(2)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_densenet(num_layers, pretrained=False, root=os.path.expanduser('~/.torch/models'),
                 **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
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
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    net = DenseNet(num_init_features, growth_rate, block_config, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('densenet%d' % num_layers, root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def densenet121(**kwargs):
    r"""Densenet-BC 121-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
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
    return get_densenet(121, **kwargs)


def densenet161(**kwargs):
    r"""Densenet-BC 161-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
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
    return get_densenet(161, **kwargs)


def densenet169(**kwargs):
    r"""Densenet-BC 169-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_densenet(169, **kwargs)


def densenet201(**kwargs):
    r"""Densenet-BC 201-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_densenet(201, **kwargs)


if __name__ == '__main__':
    net1 = densenet121()
    net2 = densenet161()
    net3 = densenet169()
    net4 = densenet201()
    import torch

    a = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        net1(a)
        net2(a)
        net3(a)
        net4(a)
