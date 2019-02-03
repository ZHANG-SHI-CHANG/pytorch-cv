import torch
from torch import nn

__all__ = ['_conv3x3', '_bn_no_affine']


def _conv3x3(in_channels, channels, stride):
    return nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)


# for mobile net
def _add_conv(out, in_channels, channels=1, kernel=1, stride=1, pad=0, num_group=1,
              active=True, relu6=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_dw(out, in_channels, dw_channels, channels, stride, relu6=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
    _add_conv(out, in_channels, dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    _add_conv(out, dw_channels, channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=None)


# for squeeze net
def _make_fire_conv(in_channels, channels, kernel_size, padding=0):
    out = list()
    out.append(nn.Conv2d(in_channels, channels, kernel_size, padding=padding))
    out.append(nn.ReLU(inplace=True))
    return nn.Sequential(*out)


class MakeFire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(MakeFire, self).__init__()
        self.s1 = _make_fire_conv(in_channels, squeeze_channels, 1)
        self.e1 = _make_fire_conv(squeeze_channels, expand1x1_channels, 1)
        self.e3 = _make_fire_conv(squeeze_channels, expand3x3_channels, 3, 1)

    def forward(self, x):
        x = self.s1(x)
        e1 = self.e1(x)
        e3 = self.e3(x)
        return torch.cat([e1, e3], 1)


# for dense net
class DenseLayer(nn.Module):
    def __init__(self, in_channel, growth_rate, bn_size, dropout, norm_layer, norm_kwargs):
        super(DenseLayer, self).__init__()
        features = list()
        features.append(norm_layer(in_channel, **({} if norm_kwargs is None else norm_kwargs)))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(in_channel, bn_size * growth_rate, kernel_size=1, bias=False))
        features.append(norm_layer(bn_size * growth_rate, **({} if norm_kwargs is None else norm_kwargs)))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        if dropout:
            features.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*features)

    def forward(self, x):
        out = self.features(x)
        return torch.cat([x, out], 1)


def _make_dense_block(in_channel, num_layers, bn_size, growth_rate, dropout,
                      norm_layer, norm_kwargs):
    out = list()
    for _ in range(num_layers):
        out.append(DenseLayer(in_channel, growth_rate, bn_size, dropout, norm_layer, norm_kwargs))
        in_channel = in_channel + growth_rate
    return nn.Sequential(*out)


def _make_transition(in_channels, num_output_features, norm_layer, norm_kwargs):
    out = list()
    out.append(norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)))
    out.append(nn.ReLU(inplace=True))
    out.append(nn.Conv2d(in_channels, num_output_features, kernel_size=1, bias=False))
    out.append(nn.AvgPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*out)


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
