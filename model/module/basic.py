import torch
from torch import nn

__all__ = ['_conv3x3', '_bn_no_affine']


def _conv3x3(in_channels, channels, stride):
    return nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)


# batch normalization affine=False: in order to fit gluon
def _bn_no_affine(channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
    bn_layer = norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs))
    nn.init.ones_(bn_layer.weight)
    nn.init.zeros_(bn_layer.bias)
    bn_layer.weight.requires_grad = False
    bn_layer.bias.requires_grad = False
    return bn_layer


# init scale: in order to fit gluon (Parameter without grad)
def _init_scale(scale=[0.229, 0.224, 0.225]):
    param = nn.Parameter(torch.Tensor(scale).view(1, 3, 1, 1) * 255, requires_grad=False)
    return param





if __name__ == '__main__':
    bn = _bn_no_affine(10)
    print(bn.weight.requires_grad)
    print(bn)
