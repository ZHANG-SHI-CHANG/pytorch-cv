import os
import torch
from torch import nn
import torch.nn.functional as F

from model.module.features import FPNFeatureExpander
from model.models_zoo.yolact import ProtoNet, PredictHead
from model.models_zoo.yolact import Detect

__all__ = ['Yolact', 'get_yolact',
           'yolact_fpn_resnet50_v1b_coco',
           'yolact_fpn_darknet53_coco',
           'yolact_fpn_resnet101_v1b_coco']


class Yolact(nn.Module):
    def __init__(self, features, src_channels, aspect_ratios, scales, use_conv_down=True,
                 num_features=256, classes=None, use_semantic_segmentation_loss=True,
                 ):
        super(Yolact, self).__init__()
        self.classes = classes
        num_classes = len(classes) + 1
        self.features = features
        if use_conv_down:
            self.down_layers = nn.ModuleList([
                nn.Conv2d(num_features, num_features, 3, 2, 1),
                nn.Conv2d(num_features, num_features, 3, 2, 1)])
        self.proto_net = ProtoNet(256, 256, 32)
        self.pred_heads = nn.ModuleList()
        for i in range(len(src_channels)):
            if i == 0:
                pred = PredictHead(src_channels[i], src_channels[i], parent=None,
                                   aspect_ratios=aspect_ratios[i], scales=scales[i])
            else:
                pred = PredictHead(src_channels[i], src_channels[i], parent=self.pred_heads[0],
                                   aspect_ratios=aspect_ratios[i], scales=scales[i])
            self.pred_heads.append(pred)
        if use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], num_classes - 1, 1)

        self.detect = Detect(num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
        self.use_conv_down = use_conv_down

    def forward(self, x):
        outs = self.features(x)
        if self.use_conv_down:
            for i, down_layer in enumerate(self.down_layers):
                outs.append(down_layer(outs[-1]))
        proto_x = outs[0]
        proto_out = self.proto_net(proto_x).permute(0, 2, 3, 1).contiguous()
        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}
        for i, pred_head in enumerate(self.pred_heads):
            p = pred_head(outs[i])
            for k, v in p.items():
                pred_outs[k].append(v)
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:
            pass
        else:
            pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

        return self.detect(pred_outs)


def get_yolact(name, dataset, pretrained=False,
               root=os.path.expanduser('~/.torch/models'), **kwargs):
    net = Yolact(**kwargs)
    if pretrained:
        from model.model_store import get_model_file
        full_name = '_'.join(('yolact', name, dataset))
        net.load_state_dict(torch.load(get_model_file(full_name, root=root)))
    return net


def yolact_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    features = FPNFeatureExpander(
        network='resnet50_v1b', outputs=[[5, 3], [6, 5], [7, 2]],
        channels=[512, 1024, 2048], num_filters=[256, 256, 256, 256],
        use_1x1=True, use_upsample=True, use_elewadd=True, use_relu=True,
        use_bias=True, version='v2', pretrained=pretrained_base)
    aspect_ratios = [[[1, 0.7071067811865475, 1.4142135623730951]]] * 5
    scales = [[24], [48], [96], [192], [384]]
    return get_yolact('fpn_resnet50_v1b', 'coco', pretrained=pretrained, classes=classes,
                      features=features, src_channels=[256] * 5, aspect_ratios=aspect_ratios,
                      scales=scales, **kwargs)


def yolact_fpn_darknet53_coco(pretrained=False, pretrained_base=True, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    features = FPNFeatureExpander(
        network='darknet53', outputs=[[14], [23], [28]],
        channels=[256, 512, 1024], num_filters=[256, 256, 256, 256],
        use_1x1=True, use_upsample=True, use_elewadd=True, use_relu=True,
        use_bias=True, version='v2', pretrained=pretrained_base)
    aspect_ratios = [[[1, 0.7071067811865475, 1.4142135623730951]]] * 5
    scales = [[24], [48], [96], [192], [384]]
    return get_yolact('fpn_darknet53', 'coco', pretrained=pretrained, classes=classes,
                      features=features, src_channels=[256] * 5, aspect_ratios=aspect_ratios,
                      scales=scales, **kwargs)


def yolact_fpn_resnet101_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    features = FPNFeatureExpander(
        network='resnet101_v1b', outputs=[[5, 3], [6, 22], [7, 2]],
        channels=[512, 1024, 2048], num_filters=[256, 256, 256, 256],
        use_1x1=True, use_upsample=True, use_elewadd=True, use_relu=True,
        use_bias=True, version='v2', pretrained=pretrained_base)
    aspect_ratios = [[[1, 0.7071067811865475, 1.4142135623730951]]] * 5
    scales = [[24], [48], [96], [192], [384]]
    return get_yolact('fpn_resnet101_v1b', 'coco', pretrained=pretrained, classes=classes,
                      features=features, src_channels=[256] * 5, aspect_ratios=aspect_ratios,
                      scales=scales, **kwargs)


if __name__ == '__main__':
    net = yolact_fpn_resnet50_v1b_coco(pretrained=True)
    net.eval()
    # print(net)

    # keys = net.state_dict().keys()
    # for key in keys:
    #     print(":\""+key+"\",")

    import numpy as np
    net.cuda()
    np.random.seed(10)
    batch = np.random.randn(1, 3, 300, 300).astype(np.float32)
    batch = torch.from_numpy(batch).cuda().float()
    with torch.no_grad():
        preds = net(batch)
    print(preds)
