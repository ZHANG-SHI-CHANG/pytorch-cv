import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['_FCNHead', '_PSPHead', '_DeepLabHead', '_DANetHead']


# for fcn
class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__(**kwargs)
        self.block = list()
        inter_channels = in_channels // 4
        self.block.append(nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False))
        self.block.append(norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(inter_channels, channels, kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


# for psp
def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    block = list()
    block.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
    block.append(norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
    block.append(nn.ReLU(inplace=True))
    return nn.Sequential(*block)


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = in_channels // 4
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def pool(self, x, size):
        return F.adaptive_avg_pool2d(x, output_size=size)

    def upsample(self, x, h, w):
        return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, h, w = x.shape
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), h, w)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), h, w)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), h, w)
        feat4 = self.upsample(self.conv4(self.pool(x, 4)), h, w)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__(**kwargs)
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)
        self.block = list()
        self.block.append(nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False))
        self.block.append(norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(512, nclass, kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


# for deeplab
def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = list()
    block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rate,
                           dilation=atrous_rate, bias=False))
    block.append(norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
    block.append(nn.ReLU(inplace=True))
    return nn.Sequential(*block)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs,
                 height=60, width=60, **kwargs):
        super(_AsppPooling, self).__init__(**kwargs)
        self.gap = list()
        self._up_kwargs = (height, width)
        self.gap.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.gap.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.gap.append(norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.gap.append(nn.ReLU(inplace=True))
        self.gap = nn.Sequential(*self.gap)

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, self._up_kwargs, mode='bilinear', align_corners=True)


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs,
                 height=60, width=60):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = list()
        self.b0.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.b0.append(norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.b0.append(nn.ReLU(inplace=True))
        self.b0 = nn.Sequential(*self.b0)

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                               norm_kwargs=norm_kwargs, height=height, width=width)

        self.project = list()
        self.project.append(nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False))
        self.project.append(norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.project.append(nn.ReLU(inplace=True))
        self.project.append(nn.Dropout(0.5))
        self.project = nn.Sequential(*self.project)

    def forward(self, x):
        a0 = self.b0(x)
        a1 = self.b1(x)
        a2 = self.b2(x)
        a3 = self.b3(x)
        a4 = self.b4(x)
        return self.project(torch.cat([a0, a1, a2, a3, a4], 1))


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer,
                          norm_kwargs=norm_kwargs, **kwargs)
        self.block = list()
        self.block.append(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.block.append(norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(256, nclass, kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


# for danet
class _PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(_PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=1)
        proj_value = self.value_conv(x).view(b, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out


class _CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(_CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(b, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out


class _DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                    nn.ReLU(inplace=True))

        self.sa = _PAM_Module(inter_channels)
        self.sc = _CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return [sasc_output, sa_output, sc_output]
