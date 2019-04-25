"""SqueezeNet, implemented in PyTorch."""

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

import os
from torch import nn
import torch.nn.functional as F

from model.module.basic import MakeFire


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class SqueezeNet(nn.Module):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self, version, classes=1000, **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        assert version in ['1.0', '1.1'], ("Unsupported SqueezeNet version {version}:"
                                           "1.0 or 1.1 expected".format(version=version))
        self.features = list()
        if version == '1.0':
            self.features.append(nn.Conv2d(3, 96, kernel_size=7, stride=2))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(96, 16, 64, 64))
            self.features.append(MakeFire(128, 16, 64, 64))
            self.features.append(MakeFire(128, 32, 128, 128))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(256, 32, 128, 128))
            self.features.append(MakeFire(256, 48, 192, 192))
            self.features.append(MakeFire(384, 48, 192, 192))
            self.features.append(MakeFire(384, 64, 256, 256))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(512, 64, 256, 256))
        else:
            self.features.append(nn.Conv2d(3, 64, kernel_size=3, stride=2))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(64, 16, 64, 64))
            self.features.append(MakeFire(128, 16, 64, 64))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(128, 32, 128, 128))
            self.features.append(MakeFire(256, 32, 128, 128))
            self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            self.features.append(MakeFire(256, 48, 192, 192))
            self.features.append(MakeFire(384, 48, 192, 192))
            self.features.append(MakeFire(384, 64, 256, 256))
            self.features.append(MakeFire(512, 64, 256, 256))
        self.features.append(nn.Dropout(0.5))
        self.features = nn.Sequential(*self.features)

        self.output = list()
        self.output.append(nn.Conv2d(512, classes, kernel_size=1))
        self.output.append(nn.ReLU(inplace=True))
        self.output = nn.Sequential(*self.output)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(3).squeeze(2)
        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_squeezenet(version, pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'),
                   **kwargs):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = SqueezeNet(version, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('squeezenet%s' % version, root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def squeezenet1_0(**kwargs):
    r"""SqueezeNet 1.0 model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.0', **kwargs)


def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.1', **kwargs)


if __name__ == '__main__':
    net1 = squeezenet1_0()
    net2 = squeezenet1_1()
    import torch
    a = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        net1(a)
        net2(a)
