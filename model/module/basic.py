import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['_conv3x3', '_bn_no_affine', 'GroupNorm']


def _conv3x3(in_channels, channels, stride):
    return nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)


# for darknet
def _conv2d(in_channel, channel, kernel, padding, stride):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.Sequential(
        nn.Conv2d(in_channel, channel, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9), nn.LeakyReLU(0.1, inplace=True)
    )
    return cell


# for inception
def _make_basic_conv(in_channel, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channel, bias=False, **kwargs),
        nn.BatchNorm2d(kwargs['out_channels']),
        nn.ReLU(inplace=True)
    )


def _make_branch(in_channel, use_pool, *conv_settings):
    out = list()
    if use_pool == 'avg':
        out.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
    elif use_pool == 'max':
        out.append(nn.MaxPool2d(kernel_size=3, stride=2))
    setting_names = ['out_channels', 'kernel_size', 'stride', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.append(_make_basic_conv(in_channel, **kwargs))
        in_channel = kwargs['out_channels']
    return nn.Sequential(*out)


class MakeA(nn.Module):
    def __init__(self, in_channel, pool_features):
        super(MakeA, self).__init__()
        self.out1 = _make_branch(in_channel, None, (64, 1, 1, 0))
        self.out2 = _make_branch(in_channel, None, (48, 1, 1, 0), (64, 5, 1, 2))
        self.out3 = _make_branch(in_channel, None, (64, 1, 1, 0), (96, 3, 1, 1), (96, 3, 1, 1))
        self.out4 = _make_branch(in_channel, 'avg', (pool_features, 1, 1, 0))

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        o4 = self.out4(x)
        # out channels = 64+64+96+pool_features
        return torch.cat([o1, o2, o3, o4], 1)


class MakeB(nn.Module):
    def __init__(self, in_channel):
        super(MakeB, self).__init__()
        self.out1 = _make_branch(in_channel, None, (384, 3, 2, 0))
        self.out2 = _make_branch(in_channel, None, (64, 1, 1, 0), (96, 3, 1, 1), (96, 3, 2, 0))
        self.out3 = _make_branch(in_channel, 'max')

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        # out_channels=384+96+in_channels
        return torch.cat([o1, o2, o3], 1)


class MakeC(nn.Module):
    def __init__(self, in_channel, channels_7x7):
        super(MakeC, self).__init__()
        self.out1 = _make_branch(in_channel, None, (192, 1, 1, 0))
        self.out2 = _make_branch(in_channel, None, (channels_7x7, 1, 1, 0),
                                 (channels_7x7, (1, 7), 1, (0, 3)), (192, (7, 1), 1, (3, 0)))
        self.out3 = _make_branch(in_channel, None, (channels_7x7, 1, 1, 0),
                                 (channels_7x7, (7, 1), 1, (3, 0)), (channels_7x7, (1, 7), 1, (0, 3)),
                                 (channels_7x7, (7, 1), 1, (3, 0)), (192, (1, 7), 1, (0, 3)))
        self.out4 = _make_branch(in_channel, 'avg', (192, 1, 1, 0))

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        o4 = self.out4(x)
        # out_channels=192*4=768
        return torch.cat([o1, o2, o3, o4], 1)


class MakeD(nn.Module):
    def __init__(self, in_channel):
        super(MakeD, self).__init__()
        self.out1 = _make_branch(in_channel, None, (192, 1, 1, 0), (320, 3, 2, 0))
        self.out2 = _make_branch(in_channel, None, (192, 1, 1, 0), (192, (1, 7), 1, (0, 3)),
                                 (192, (7, 1), 1, (3, 0)), (192, 3, 2, 0))
        self.out3 = _make_branch(in_channel, 'max')

    def forward(self, x):
        o1 = self.out1(x)
        o2 = self.out2(x)
        o3 = self.out3(x)
        # out_channels=192*2+in_channels
        return torch.cat([o1, o2, o3], 1)


class MakeE(nn.Module):
    def __init__(self, in_channel):
        super(MakeE, self).__init__()
        self.s0 = _make_branch(in_channel, None, (320, 1, 1, 0))

        self.s1 = _make_branch(in_channel, None, (384, 1, 1, 0))
        self.s11 = _make_branch(384, None, (384, (1, 3), 1, (0, 1)))
        self.s12 = _make_branch(384, None, (384, (3, 1), 1, (1, 0)))

        self.s2 = _make_branch(in_channel, None, (448, 1, 1, 0),
                               (384, 3, 1, 1))
        self.s21 = _make_branch(384, None, (384, (1, 3), 1, (0, 1)))
        self.s22 = _make_branch(384, None, (384, (3, 1), 1, (1, 0)))

        self.s3 = _make_branch(in_channel, 'avg', (192, 1, 1, 0))

    def forward(self, x):
        o0 = self.s0(x)
        o1 = self.s1(x)
        o11 = self.s11(o1)
        o12 = self.s12(o1)
        o1 = torch.cat([o11, o12], 1)
        o2 = self.s2(x)
        o21 = self.s21(o2)
        o22 = self.s22(o2)
        o2 = torch.cat([o21, o22], 1)
        o3 = self.s3(x)
        # out_channels=384*4+192
        return torch.cat([o0, o1, o2, o3], 1)


# for mobile net
def _add_conv(out, in_channels, channels=1, kernel=1, stride=1, pad=0, num_group=1,
              active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_dw(out, in_channels, dw_channels, channels, stride, relu6=False):
    _add_conv(out, in_channels, dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, dw_channels, channels, relu6=relu6)


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
    def __init__(self, in_channel, growth_rate, bn_size, dropout):
        super(DenseLayer, self).__init__()
        features = list()
        features.append(nn.BatchNorm2d(in_channel))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(in_channel, bn_size * growth_rate, kernel_size=1, bias=False))
        features.append(nn.BatchNorm2d(bn_size * growth_rate))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        if dropout:
            features.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*features)

    def forward(self, x):
        out = self.features(x)
        return torch.cat([x, out], 1)


def _make_dense_block(in_channel, num_layers, bn_size, growth_rate, dropout):
    out = list()
    for _ in range(num_layers):
        out.append(DenseLayer(in_channel, growth_rate, bn_size, dropout))
        in_channel = in_channel + growth_rate
    return nn.Sequential(*out)


def _make_transition(in_channels, num_output_features):
    out = nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, num_output_features, kernel_size=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return out


# batch normalization affine=False: in order to fit gluon
def _bn_no_affine(channels):
    bn_layer = nn.BatchNorm2d(channels)
    nn.init.ones_(bn_layer.weight)
    nn.init.zeros_(bn_layer.bias)
    bn_layer.weight.requires_grad = False
    bn_layer.bias.requires_grad = False
    return bn_layer


# init scale: in order to fit gluon (Parameter without grad)
def _init_scale(scale=[0.229, 0.224, 0.225]):
    param = nn.Parameter(torch.Tensor(scale).view(1, 3, 1, 1) * 255, requires_grad=False)
    return param


# group norm with default num_groups
class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)


# for hourglass
class BasicConv(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True, with_relu=True):
        super(BasicConv, self).__init__()
        p = (k - 1) // 2
        self.with_bn, self.with_relu = with_bn, with_relu
        self.conv = nn.Conv2d(inp_dim, out_dim, k, stride, p, bias=not with_bn)
        if with_bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return F.relu(x) if self.with_relu else x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(Residual, self).__init__()
        p = (k - 1) // 2

        self.feat = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, k, stride, p, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, k, 1, p, bias=False),
            nn.BatchNorm2d(out_dim))

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, 1, stride, bias=False),
            nn.BatchNorm2d(out_dim),
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()

    def forward(self, x):
        out = self.feat(x)

        skip = self.skip(x)
        return F.relu(out + skip)


class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class Merge(nn.Module):
    def forward(self, x, y):
        return x + y


def _layer(inp_dim, out_dim, num):
    layers = [Residual(inp_dim, out_dim)]
    layers += [Residual(out_dim, out_dim) for _ in range(1, num)]
    return nn.Sequential(*layers)


def _layer_reverse(inp_dim, out_dim, num):
    layers = [Residual(inp_dim, inp_dim) for _ in range(num - 1)]
    layers += [Residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)


def _pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def _unpool_layer(dim):
    return Upsample(scale_factor=2)


def _merge_layer(dim):
    return Merge()


if __name__ == '__main__':
    # bn = _bn_no_affine(10)
    # print(bn.weight.requires_grad)
    # print(bn)
    m = GroupNorm(6)
    print(m)
