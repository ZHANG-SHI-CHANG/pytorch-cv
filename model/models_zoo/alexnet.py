"""Alexnet, implemented in PyTorch."""

__all__ = ['AlexNet', 'alexnet']

import os
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Net
# -----------------------------------------------------------------------------
class AlexNet(nn.Module):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, classes=1000, img_size=224, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.features = list()
        self.features.append(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.append(nn.Conv2d(64, 192, kernel_size=5, padding=2))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.append(nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features = nn.Sequential(*self.features)

        self.liner = list()
        self.liner.append(nn.Linear(256 * (((img_size - 7) // 4 + 1) // 8) ** 2, 4096))
        self.liner.append(nn.ReLU(inplace=True))
        self.liner.append(nn.Linear(4096, 4096))
        self.liner.append(nn.ReLU(inplace=True))
        self.liner.append(nn.Dropout(0.5))
        self.liner = nn.Sequential(*self.liner)

        self.output = nn.Linear(4096, classes)

    def forward(self, x):
        x = self.features(x).view(x.shape[0], -1)
        x = self.liner(x)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def alexnet(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = AlexNet(**kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('alexnet', root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


if __name__ == '__main__':
    import torch

    a = torch.randn(2, 3, 224, 224)
    net = alexnet()
    print(net)
    with torch.no_grad():
        net(a)
