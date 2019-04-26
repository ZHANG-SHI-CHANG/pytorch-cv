# TODO: unfinish
import os
import math
import torch
from torch import nn
import torch.nn.functional as F

from model.models_zoo.seg.segbase import SegBaseModel
from model.module.basic import _make_basic_conv
from model.module.features import _parse_network

__all__ = ['get_bisenet', 'get_bisenet_resnet18_voc',
           'get_bisenet_resnet18_citys']


# module
class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = _make_basic_conv(in_planes, out_channels=inner_channel,
                                         kernel_size=7, stride=2, padding=3)
        self.conv_3x3_1 = _make_basic_conv(inner_channel, out_channels=inner_channel,
                                           kernel_size=3, stride=2, padding=1)
        self.conv_3x3_2 = _make_basic_conv(inner_channel, out_channels=inner_channel,
                                           kernel_size=3, stride=2, padding=1)
        self.conv_1x1 = _make_basic_conv(inner_channel, out_channels=out_planes,
                                         kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = _make_basic_conv(in_planes, out_channels=out_planes,
                                         kernel_size=3, stride=1, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _make_basic_conv(out_planes, out_channels=out_planes,
                             kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = torch.sigmoid(self.channel_attention(fm))
        fm = fm * fm_se

        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = _make_basic_conv(in_planes, out_channels=out_planes,
                                         kernel_size=1, stride=1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, out_planes // reduction, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // reduction, out_planes, 1, 1, 0)
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = torch.sigmoid(self.channel_attention(fm))
        output = fm + fm * fm_se
        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, size, is_aux=False):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = _make_basic_conv(in_planes, out_channels=256,
                                             kernel_size=3, stride=1, padding=1)
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_3x3 = _make_basic_conv(in_planes, out_channels=64,
                                             kernel_size=3, stride=1, padding=1)
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)

        self.size = size

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        output = F.interpolate(output, size=self.size,
                               mode='bilinear', align_corners=True)

        return output


class BiseNet(SegBaseModel):
    def __init__(self, nclass, out_planes, backbone=None, aux=False, dilated=False, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(BiseNet, self).__init__(nclass, aux, backbone, base_size=base_size, crop_size=crop_size,
                                      dilated=dilated, jpu=jpu, pretrained_base=pretrained_base, **kwargs)
        assert not dilated and not jpu
        conv_channel = 128
        self.spatial_path = SpatialPath(3, 128)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _make_basic_conv(512, out_channels=conv_channel, kernel_size=1, stride=1, padding=0)
        )

        self.arms = nn.ModuleList([AttentionRefinement(512, conv_channel),
                                   AttentionRefinement(256, conv_channel)])
        self.refines = nn.ModuleList([
            _make_basic_conv(conv_channel, out_channels=conv_channel, kernel_size=3, stride=1, padding=1),
            _make_basic_conv(conv_channel, out_channels=conv_channel, kernel_size=3, stride=1, padding=1),
        ])

        self.heads = nn.ModuleList([
            BiSeNetHead(conv_channel, out_planes, self._up_kwargs, True),
            BiSeNetHead(conv_channel, out_planes, self._up_kwargs, True),
            BiSeNetHead(conv_channel * 2, out_planes, self._up_kwargs, False)
        ])

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 4)

        self.__setattr__('others', ['spatial_path', 'global_context', 'arms', 'refines', 'heads'])

    def forward(self, x):
        spatial_out = self.spatial_path(x)

        context_outs = list(self.base_forward(x))
        context_outs.reverse()

        global_context = self.global_context(context_outs[0])
        global_context = F.interpolate(global_context, size=context_outs[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = list()
        for i, (fm, arm, refine) in enumerate(zip(context_outs[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(math.ceil(self._up_kwargs[0] / (2 ** (4 - i))),
                                              math.ceil(self._up_kwargs[1] / (2 ** (4 - i)))),  # TODO: may have bug
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        concat_fm = self.ffm(spatial_out, last_fm)
        pred_out.append(concat_fm)
        if self.training:
            out_list = list()
            for pred, head in zip(pred_out, self.heads):
                out_list.append(head(pred))
            return out_list
        else:
            self.heads[-1].size = self._up_kwargs
            return [F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)]


bisenet_spec = {
    # out_planes, backbone, outputs
    'resnet18': [19, 'resnet18_v1b', [[5, 1], [6, 1], [7, 1]], ]
}


def get_bisenet(dataset='pascal_paper', backbone='resnet18', pretrained_base=True,
                pretrained=False, root=os.path.expanduser('~/.torch/models'), **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_paper': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data import datasets
    config = bisenet_spec[backbone]
    if config[2] is not None:
        feat = nn.ModuleList(_parse_network(config[1], outputs=config[2], pretrained=pretrained_base))
    else:
        feat = config[1]
    model = BiseNet(config[0], datasets[dataset].NUM_CLASS, backbone=feat, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('bisenet_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_bisenet_resnet18_voc(**kwargs):
    return get_bisenet('pascal_paper', 'resnet18', **kwargs)


def get_bisenet_resnet18_citys(**kwargs):
    return get_bisenet('citys', 'resnet18', **kwargs)


if __name__ == '__main__':
    net = get_bisenet_resnet18_voc()
    net.eval()
    a = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        out = net(a)
    print([k.shape for k in out])
