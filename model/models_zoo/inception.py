"""Inception, implemented in PyTorch."""

__all__ = ['Inception3', 'inception_v3']

import os
from torch import nn

from model.module.basic import _make_basic_conv
from model.module.basic import MakeA, MakeB, MakeC, MakeD, MakeE


def make_aux(classes, norm_layer, norm_kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(channels=128, kernel_size=1,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    out.add(_make_basic_conv(channels=768, kernel_size=5,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    out.add(nn.Flatten())
    out.add(nn.Dense(classes))
    return out


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class Inception3(nn.Module):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, classes=1000, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(Inception3, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        self.features = list()
        self.features.append(_make_basic_conv(3, out_channels=32, kernel_size=3, stride=2,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.features.append(_make_basic_conv(32, out_channels=32, kernel_size=3,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.features.append(_make_basic_conv(32, out_channels=64, kernel_size=3, padding=1,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.append(_make_basic_conv(64, out_channels=80, kernel_size=1,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.features.append(_make_basic_conv(80, out_channels=192, kernel_size=3,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.append(MakeA(192, 32, norm_layer, norm_kwargs))
        self.features.append(MakeA(224 + 32, 64, norm_layer, norm_kwargs))
        self.features.append(MakeA(224 + 64, 64, norm_layer, norm_kwargs))
        self.features.append(MakeB(224 + 64, norm_layer, norm_kwargs))
        self.features.append(MakeC(768, 128, norm_layer, norm_kwargs))
        self.features.append(MakeC(768, 160, norm_layer, norm_kwargs))
        self.features.append(MakeC(768, 160, norm_layer, norm_kwargs))
        self.features.append(MakeC(768, 192, norm_layer, norm_kwargs))
        self.features.append(MakeD(768, norm_layer, norm_kwargs))
        self.features.append(MakeE(1280, norm_layer, norm_kwargs))
        self.features.append(MakeE(2048, norm_layer, norm_kwargs))
        self.features.append(nn.AvgPool2d(kernel_size=8))
        self.features.append(nn.Dropout(0.5))
        self.features = nn.Sequential(*self.features)

        self.output = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.features(x).squeeze(3).squeeze(2)
        x = self.output(x)
        return x


# Constructor
def inception_v3(pretrained=False, root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """
    net = Inception3(**kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('inceptionv3', root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


if __name__ == '__main__':
    net = inception_v3()
    # print(net)
    import torch

    a = torch.randn(2, 3, 299, 299)
    with torch.no_grad():
        out = net(a)
    print(out.shape)
