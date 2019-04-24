import os
from torch import nn

from model.module.features import _parse_network
from model.models_zoo.centernet import ResDeConvLayer, DLADeConvLayer, HeadBranch

__all__ = ['get_centernet',
           'centernet_resnet18_dcn_coco',
           'centernet_resnet101_dcn_coco',
           'centernet_dla34_dcn_coco']


class CenterNet(nn.Module):
    def __init__(self, features, deconv_layers, heads, head_conv, classes,
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.features = features
        self.deconv_layers = deconv_layers
        self.classes = classes
        self.head_branchs = HeadBranch(heads, head_conv)

    def forward(self, x):
        if isinstance(self.features, nn.ModuleList):
            out = list()
            for feat in self.features:
                x = feat(x)
                out.append(x)
            x = out
        else:
            x = self.features(x)
        x = self.deconv_layers(x)
        return self.head_branchs(x)


def get_centernet(name, features, deconv_layers, heads, head_conv, pretrained=False,
                  root=os.path.expanduser('~/.torch/models'), **kwargs):
    net = CenterNet(features, deconv_layers, heads, head_conv, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(name, root=root)))
    return net


def centernet_resnet18_dcn_coco(pretrained=False, pretrained_base=True, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    name = 'centernet_resnet18_dcn_coco'
    from model.models_zoo.resnet import resnet18_v1
    norm_layer = kwargs.get('norm_layer') if 'norm_layer' in kwargs else nn.BatchNorm2d
    norm_kwargs = kwargs.get('norm_kwargs') if 'norm_kwargs' in kwargs else None
    features = resnet18_v1(pretrained=pretrained_base, norm_layer=norm_layer,
                           norm_kwargs=norm_kwargs).features
    deconv_layers = ResDeConvLayer(512, 3, [256, 128, 64], [4, 4, 4],
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    return get_centernet(name, features, deconv_layers, heads={'hm': 80, 'wh': 2, 'reg': 2}, head_conv=64,
                         classes=classes, pretrained=pretrained, **kwargs)


def centernet_resnet101_dcn_coco(pretrained=False, pretrained_base=False, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    name = 'centernet_resnet101_dcn_coco'
    from model.models_zoo.resnet_ori import resnet101_v1
    norm_layer = kwargs.get('norm_layer') if 'norm_layer' in kwargs else nn.BatchNorm2d
    norm_kwargs = kwargs.get('norm_kwargs') if 'norm_kwargs' in kwargs else None
    features = resnet101_v1(pretrained=pretrained_base, norm_layer=norm_layer,
                            norm_kwargs=norm_kwargs).features
    deconv_layers = ResDeConvLayer(2048, 3, [256, 128, 64], [4, 4, 4],
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    return get_centernet(name, features, deconv_layers, heads={'hm': 80, 'wh': 2, 'reg': 2}, head_conv=64,
                         classes=classes, pretrained=pretrained, **kwargs)


def centernet_dla34_dcn_coco(pretrained=False, pretrained_base=False, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    name = 'centernet_dla34_dcn_coco'
    norm_layer = kwargs.get('norm_layer') if 'norm_layer' in kwargs else nn.BatchNorm2d
    norm_kwargs = kwargs.get('norm_kwargs') if 'norm_kwargs' in kwargs else None
    outputs = [[1], [2], [3], [4], [5], [6]]
    features = nn.ModuleList(_parse_network('dla34', outputs, pretrained_base, norm_layer=norm_layer,
                                            norm_kwargs=norm_kwargs))
    deconv_layers = DLADeConvLayer([16, 32, 64, 128, 256, 512], down_ratio=4, last_level=5,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    return get_centernet(name, features, deconv_layers, heads={'hm': 80, 'wh': 2, 'reg': 2}, head_conv=256,
                         classes=classes, pretrained=pretrained, **kwargs)


if __name__ == '__main__':
    import torch

    net = centernet_dla34_dcn_coco(pretrained=True).cuda()
    net.eval()
    # print(net)

    import numpy as np

    np.random.seed(10)
    a = np.random.randn(1, 3, 224, 224).astype(np.float32)
    images = torch.from_numpy(a).cuda()
    with torch.no_grad():
        out = net(images)
    print(out)
    print([k.shape for k in out.values()])

    # print(net)

    # for key in net.state_dict().keys():
    #     print(":\"" + key + "\", ")
    # print(len(net.state_dict().keys()))
