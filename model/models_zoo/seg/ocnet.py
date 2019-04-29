import os
import torch
from torch import nn
import torch.nn.functional as F

from model.models_zoo.seg import SegBaseModel
from model.module.basic_seg import _FCNHead
from model.module.oc_block import BaseOCModule, ASPOCModule

__all__ = ['get_ocnet', 'get_ocnet_asp_resnet101_voc', 'get_ocnet_base_resnet101_voc']


class OCNet(SegBaseModel):
    def __init__(self, nclass, oc='base', backbone='resnet101', aux=False, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(OCNet, self).__init__(nclass, aux, backbone, dilated=dilated, jpu=jpu, base_size=base_size,
                                    crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.context = self.get_oc(oc)
        self.cls = nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass)

        self.__setattr__('others', ['cls', 'context', 'auxlayer'] if aux else ['cls', 'context'])

    @staticmethod
    def get_oc(oc):
        if oc == 'base':
            return nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                BaseOCModule(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
                             dropout=0.05, sizes=([1]))
            )
        elif oc == 'asp':
            return nn.Sequential(
                ASPOCModule(2048, 512))
        else:
            raise ValueError('illegal type')

    def forward(self, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.context(c4)
        x = self.cls(x)
        x = F.interpolate(x, self._up_kwargs, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, self._up_kwargs, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return outputs


def get_ocnet(dataset='pascal_voc', backbone='resnet50', oc='base', pretrained=False, pretrained_base=True,
              root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_paper': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data import datasets
    # infer number of classes
    model = OCNet(datasets[dataset].NUM_CLASS, oc=oc, backbone=backbone,
                  pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('ocnet_%s_%s_%s' % (oc, backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_ocnet_base_resnet101_voc(**kwargs):
    return get_ocnet('pascal_paper', 'resnet101', 'base', **kwargs)


def get_ocnet_asp_resnet101_voc(**kwargs):
    return get_ocnet('pascal_paper', 'resnet101', 'asp', **kwargs)
