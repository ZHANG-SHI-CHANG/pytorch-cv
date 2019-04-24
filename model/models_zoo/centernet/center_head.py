import math
import numpy as np
from torch import nn

from model.module.dcn_v2 import DCN


# for weight init
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


# for resnet-dcn
class ResDeConvLayer(nn.Module):
    def __init__(self, in_planes, num_layers, num_filters, num_kernels, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, use_bias=False):
        super(ResDeConvLayer, self).__init__()
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = list()
        for i in range(num_layers):
            kernel, pad, out_pad = self.get_config(num_kernels[i])
            planes = num_filters[i]
            fc = DCN(in_planes, planes, kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(planes, planes, kernel, stride=2, padding=pad,
                                    output_padding=out_pad, bias=use_bias)
            fill_up_weights(up)
            layers.append(fc)
            layers.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs)))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def get_config(kernel):
        if kernel == 4:
            pad, out_pad = 1, 0
        elif kernel == 3:
            pad, out_pad = 1, 1
        elif kernel == 2:
            pad, out_pad = 0, 0
        else:
            raise ValueError('illegal kernel size')
        return kernel, pad, out_pad


class HeadBranch(nn.Module):
    def __init__(self, heads, head_conv=0):
        super(HeadBranch, self).__init__()
        self.heads = heads
        for head, num_out in heads.items():
            fc = list()
            if head_conv > 0:
                fc.append(nn.Conv2d(64, head_conv, 3, 1, 1, bias=True))
                fc.append(nn.ReLU(inplace=True))
            fc.append(nn.Conv2d(head_conv if head_conv > 0 else 64, num_out, 1, 1, 0, bias=True))
            if head == 'hm':
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weight(fc[-1])
            fc = nn.Sequential(*fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret

    @staticmethod
    def fill_fc_weight(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# for dla-dcn version
class DeformConv(nn.Module):
    def __init__(self, chi, cho, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            norm_layer(cho, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o, norm_layer, norm_kwargs)
            node = DeformConv(o, o, norm_layer, norm_kwargs)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2,
                                    output_padding=0, groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j],
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLADeConvLayer(nn.Module):
    def __init__(self, channels, down_ratio, last_level, out_channel=0,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DLADeConvLayer, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)],
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def forward(self, x):
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y[-1]
