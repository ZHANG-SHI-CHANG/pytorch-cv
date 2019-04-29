import os
import torch
import torch.nn.functional as F

from model.models_zoo.seg import SegBaseModel
from model.module.basic_seg import _FCNHead
from model.module.ca_block import RCCAModule

__all__ = ['get_ccnet', 'get_ccnet_resnet101_voc']


class CCNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet101', aux=False, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(CCNet, self).__init__(nclass, aux, backbone, dilated=dilated, jpu=jpu, base_size=base_size,
                                    crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.head = RCCAModule(2048, 512, nclass)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass)

        self.__setattr__('others', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, self._up_kwargs, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, self._up_kwargs, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return outputs


def get_ccnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, pretrained_base=True,
              root=os.path.expanduser('~/.torch/models'), **kwargs):
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
    model = CCNet(datasets[dataset].NUM_CLASS, backbone=backbone,
                  pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('ccnet_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_ccnet_resnet101_voc(**kwargs):
    return get_ccnet('pascal_paper', 'resnet101', **kwargs)


if __name__ == '__main__':
    net = get_ccnet_resnet101_voc().cuda()
    a = torch.randn(1, 3, 480, 480).cuda()
    print(net(a))
