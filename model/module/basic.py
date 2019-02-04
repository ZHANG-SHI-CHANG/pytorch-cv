import torch
from torch import nn

__all__ = ['_conv3x3', '_bn_no_affine']


def _conv3x3(in_channels, channels, stride):
    return nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)


# for darknet
def _conv2d(in_channel, channel, kernel, padding, stride, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = list()
    cell.append(nn.Conv2d(in_channel, channel, kernel_size=kernel,
                          stride=stride, padding=padding, bias=False))
    cell.append(norm_layer(channel, eps=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*cell)


# for inception
def _make_basic_conv(in_channel, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
    out = list()
    out.append(nn.Conv2d(in_channel, bias=False, **kwargs))
    out.append(norm_layer(kwargs['out_channels'], eps=0.001, **({} if norm_kwargs is None else norm_kwargs)))
    out.append(nn.ReLU(inplace=True))
    return nn.Sequential(*out)


def _make_branch(in_channel, use_pool, norm_layer, norm_kwargs, *conv_settings):
    out = list()
    if use_pool == 'avg':
        out.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
    elif use_pool == 'max':
        out.append(nn.MaxPool2D(kernel_size=3, stride=2))
    setting_names = ['out_channels', 'kernel_size', 'stride', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.append(_make_basic_conv(in_channel, norm_layer, norm_kwargs, **kwargs))
        in_channel = kwargs['out_channels']
    return nn.Sequential(*out)


class MakeA(nn.Module):
    def __init__(self, in_channel, pool_features, norm_layer, norm_kwargs):
        super(MakeA, self).__init__()
        self.out1 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (64, 1, 1, 0))
        self.out2 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (48, 1, 1, 0), (64, 5, 1, 2))
        self.out3 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (64, 1, 1, 0), (96, 3, 1, 1), (96, 3, 1, 1))
        self.out4 = _make_branch(in_channel, 'avg', norm_layer, norm_kwargs,
                                 (pool_features, 1, 1, 0))

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        o4 = self.out4(x)
        return torch.cat([o1, o2, o3, o4], 1)


class MakeB(nn.Module):
    def __init__(self, in_channel, norm_layer, norm_kwargs):
        super(MakeB, self).__init__()
        self.out1 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (384, 3, 2, 0))
        self.out2 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (64, 1, 1, 0), (96, 3, 1, 1), (96, 3, 2, 0))
        self.out3 = _make_branch(in_channel, 'max', norm_layer, norm_kwargs)

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        return torch.cat([o1, o2, o3], 1)


class MakeC(nn.Module):
    def __init__(self, in_channel, channels_7x7, norm_layer, norm_kwargs):
        super(MakeC, self).__init__()
        self.out1 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (192, 1, 1, 0))
        self.out2 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (channels_7x7, 1, 1, 0), (channels_7x7, (1, 7), 1, (0, 3)),
                                 (192, (7, 1), 1, (3, 0)))
        self.out3 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (channels_7x7, 1, 1, 0), (channels_7x7, (7, 1), 1, (3, 0)),
                                 (channels_7x7, (1, 7), 1, (0, 3)), (channels_7x7, (7, 1), 1, (3, 0)),
                                 (192, (1, 7), 1, (0, 3)))
        self.out4 = _make_branch(in_channel, 'avg', norm_layer, norm_kwargs,
                                 (192, 1, 1, 0))

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        o4 = self.out4(x)
        return torch.cat([o1, o2, o3, o4], 1)


class MakeD(nn.Module):
    def __init__(self, in_channel, norm_layer, norm_kwargs):
        super(MakeD, self).__init__()
        self.out1 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (192, 1, 1, 0), (320, 3, 2, 0))
        self.out2 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
                                 (192, 1, 1, 0), (192, (1, 7), 1, (0, 3)),
                                 (192, (7, 1), 1, (3, 0)), (192, 3, 2, 0))
        self.out3 = _make_branch(in_channel, 'max', norm_layer, norm_kwargs)

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        return torch.cat([o1, o2, o3], 1)

# class MakeE(nn.Module):
#     def __init__(self, in_channel, norm_layer, norm_kwargs):
#         super(MakeE, self).__init__()
#         self.out1 = _make_branch(in_channel, None, norm_layer, norm_kwargs,
#                                  (320, 1, 1, 0))
#
#
#     def forward(self, x):
#         o1 = self.out1(x)
#         o2 = self.out2(x)
#         o3 = self.out3(x)
#         return torch.cat([o1, o2, o3], 1)

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
