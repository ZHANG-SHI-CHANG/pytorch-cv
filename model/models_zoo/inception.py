"""Inception, implemented in PyTorch."""

__all__ = ['Inception3', 'inception_v3']

import os
from torch import nn

from model.module.basic import _make_basic_conv
from model.module.basic import MakeA, MakeB, MakeC, MakeD, MakeE


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

    def __init__(self, classes=1000, **kwargs):
        super(Inception3, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        self.features = nn.Sequential(
            _make_basic_conv(3, out_channels=32, kernel_size=3, stride=2),
            _make_basic_conv(32, out_channels=32, kernel_size=3),
            _make_basic_conv(32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _make_basic_conv(64, out_channels=80, kernel_size=1),
            _make_basic_conv(80, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MakeA(192, 32), MakeA(224 + 32, 64),
            MakeA(224 + 64, 64), MakeB(224 + 64),
            MakeC(768, 128), MakeC(768, 160),
            MakeC(768, 160), MakeC(768, 192),
            MakeD(768), MakeE(1280), MakeE(2048),
            nn.AvgPool2d(kernel_size=8), nn.Dropout(0.5)
        )

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
