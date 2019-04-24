import os
import torch
import torch.nn.functional as F

from model.models_zoo.seg.segbase import SegBaseModel
from model.module.basic_seg import _DANetHead

__all__ = ['get_danet', 'get_danet_resnet101_voc']


class DANet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet101', aux=False, pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, base_size=base_size, crop_size=crop_size,
                                    pretrained_base=pretrained_base, **kwargs)
        self.head = _DANetHead(2048, nclass, **kwargs)

    def forward(self, x):
        c3, c4 = self.base_forward(x)
        x = self.head(c4)
        x = [F.interpolate(a, self._up_kwargs, mode='bilinear', align_corners=True) for a in x]
        return x


def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_paper': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
    }
    from data import datasets
    # infer number of classes
    model = DANet(datasets[dataset].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('danet_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_danet_resnet101_voc(**kwargs):
    return get_danet('pascal_voc', 'resnet101', **kwargs)


if __name__ == '__main__':
    net = get_danet_resnet101_voc()
    net.eval()
    a = torch.randn(1, 3, 200, 200)
    with torch.no_grad():
        out = net(a)
    print(out)
