"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import warnings
import torch
from torch import nn
import torch.nn.functional as F

from model.ops import box_nms, roi_pool, roi_align
from model.module.features import FPNFeatureExpander
from model.models_zoo.rcnn import RCNN
from model.models_zoo.faster_rcnn import RCNNTargetGenerator, RCNNTargetSampler
from model.models_zoo.rpn import RPN

__all__ = ['FasterRCNN', 'get_faster_rcnn',
           'faster_rcnn_resnet50_v1b_voc', 'faster_rcnn_resnet50_v1b_coco', ]


# __all__ = ['FasterRCNN', 'get_faster_rcnn',
#            'faster_rcnn_resnet50_v1b_voc',
#            'faster_rcnn_resnet50_v1b_coco',
#            'faster_rcnn_fpn_resnet50_v1b_coco',
#            'faster_rcnn_fpn_bn_resnet50_v1b_coco',
#            'faster_rcnn_resnet50_v1b_custom',
#            'faster_rcnn_resnet101_v1d_voc',
#            'faster_rcnn_resnet101_v1d_coco',
#            'faster_rcnn_fpn_resnet101_v1d_coco',
#            'faster_rcnn_resnet101_v1d_custom']

class FasterRCNN(RCNN):
    def __init__(self, features, top_features, classes, box_features=None,
                 short=600, max_size=1000, min_stage=4, max_stage=4, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100, roi_mode='align',
                 roi_size=(14, 14), strides=16, clip=None, channel=2048,
                 rpn_in_channel=1024, rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=300,
                 additional_output=False, **kwargs):
        super(FasterRCNN, self).__init__(
            features=features, top_features=top_features, classes=classes,
            box_features=box_features, short=short, max_size=max_size,
            train_patterns=train_patterns, nms_thresh=nms_thresh, nms_topk=nms_topk,
            post_nms=post_nms, roi_mode=roi_mode, roi_size=roi_size, strides=strides, clip=clip,
            channel=channel, **kwargs)
        self.ashape = alloc_size[0]
        self._min_stage = min_stage
        self._max_stage = max_stage
        self.num_stages = max_stage - min_stage + 1
        if self.num_stages > 1:
            assert len(scales) == len(strides) == self.num_stages, \
                "The num_stages (%d) must match number of scales (%d) and strides (%d)" \
                % (self.num_stages, len(scales), len(strides))
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        self._target_generator = {RCNNTargetGenerator(self.num_class)}
        self._additional_output = additional_output
        self.rpn = RPN(
            in_channels=rpn_in_channel, channels=rpn_channel, strides=strides,
            base_size=base_size, scales=scales, ratios=ratios, alloc_size=alloc_size,
            clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
            train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
            test_post_nms=rpn_test_post_nms, min_size=rpn_min_size,
            multi_level=self.num_stages > 1)
        self.sampler = RCNNTargetSampler(
            num_image=self._max_batch, num_proposal=rpn_train_post_nms,
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            pos_ratio=pos_ratio, max_num_gt=max_num_gt)

    @property
    def target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return list(self._target_generator)[0]

    def reset_class(self, classes, reuse_weights=None):
        # TODO
        super(FasterRCNN, self).reset_class(classes, reuse_weights)
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

    def _pyramid_roi_feats(self, features, rpn_rois, roi_size, strides, roi_mode='align',
                           eps=1e-6):
        """Assign rpn_rois to specific FPN layers according to its area
           and then perform `ROIPooling` or `ROIAlign` to generate final
           region proposals aggregated features.
        Parameters
        ----------
        features : list of mx.ndarray or mx.symbol
            Features extracted from FPN base network
        rpn_rois : mx.ndarray or mx.symbol
            (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
        roi_size : tuple
            The size of each roi with regard to ROI-Wise operation
            each region proposal will be roi_size spatial shape.
        strides : tuple e.g. [4, 8, 16, 32]
            Define the gap that ori image and feature map have
        roi_mode : str, default is align
            ROI pooling mode. Currently support 'pool' and 'align'.
        Returns
        -------
        Pooled roi features aggregated according to its roi_level
        """
        max_stage = self._max_stage
        if self._max_stage > 5:  # do not use p6 for RCNN
            max_stage = self._max_stage - 1
        _, x1, y1, x2, y2 = torch.split(rpn_rois, 1, dim=-1)  # TODO
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        roi_level = torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224.0 + eps))
        roi_level = torch.clamp(roi_level, self._min_stage, max_stage).squeeze()
        # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
        # roi_level_sorted_args = F.argsort(roi_level, is_ascend=True)
        # roi_level = F.sort(roi_level, is_ascend=True)
        # rpn_rois = F.take(rpn_rois, roi_level_sorted_args, axis=0)
        pooled_roi_feats = []
        for i, l in enumerate(range(self._min_stage, max_stage + 1)):
            # Pool features with all rois first, and then set invalid pooled features to zero,
            # at last ele-wise add together to aggregate all features.
            if roi_mode == 'pool':
                pooled_feature = roi_pool(features[i], rpn_rois, roi_size, 1. / strides[i])
            elif roi_mode == 'align':
                pooled_feature = roi_align(features[i], rpn_rois, roi_size, 1. / strides[i],
                                           sampling_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(roi_mode))
            pooled_feature = torch.where((roi_level == l).view(-1, 1, 1, 1),
                                         pooled_feature, torch.zeros_like(pooled_feature))
            pooled_roi_feats.append(pooled_feature)
        # Ele-wise add to aggregate all pooled features
        pooled_roi_feats = torch.stack(pooled_roi_feats).sum(0)
        # Sort all pooled features by asceding order
        # [2,2,..,3,3,...,4,4,...,5,5,...]
        # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
        # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
        return pooled_roi_feats

    def forward(self, x, gt_box=None):
        def _split(x, axis, num_outputs, squeeze_axis):
            x = torch.split(x, x.shape[axis] // num_outputs, axis)
            x = [a.squeeze_(axis) for a in x] if squeeze_axis else [a for a in x]
            if isinstance(x, list):
                return x
            else:
                return [x]

        feat = self.features(x)
        if not isinstance(feat, (list, tuple)):
            feat = [feat]

        # RPN proposals
        if self.training:
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = \
                self.rpn(torch.zeros_like(x), *feat)
            rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_box)
        else:
            _, rpn_box = self.rpn(torch.zeros_like(x), *feat)

        # create batchid for roi
        num_roi = self._num_sample if self.training else self._rpn_test_post_nms

        # roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
        roi_batchid = torch.arange(0, self._max_batch, dtype=x.dtype, device=x.device)
        roi_batchid = roi_batchid.repeat(num_roi)
        # remove batch dim because ROIPooling require 2d input
        rpn_roi = torch.cat([roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)
        rpn_roi.detach_()

        if self.num_stages > 1:
            # using FPN
            pooled_feat = self._pyramid_roi_feats(feat, rpn_roi, self._roi_size,
                                                  self._strides, roi_mode=self._roi_mode)
        else:
            # ROI features
            if self._roi_mode == 'pool':
                pooled_feat = roi_pool(feat[0], rpn_roi, self._roi_size, 1. / self._strides)
            elif self._roi_mode == 'align':
                pooled_feat = roi_align(feat[0], rpn_roi, self._roi_size,
                                        1. / self._strides, sampling_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        if self.top_features is not None:
            top_feat = self.top_features(pooled_feat)
        else:
            top_feat = pooled_feat
        if self.box_features is None:
            box_feat = F.adaptive_avg_pool2d(top_feat, output_size=1).squeeze()
        else:
            top_feat = top_feat.reshape(top_feat.shape[0], -1)
            box_feat = self.box_features(top_feat).squeeze()
        cls_pred = self.class_predictor(box_feat)
        box_pred = self.box_predictor(box_feat)
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_class, 4))

        # no need to convert bounding boxes in training, just return
        if self.training:
            if self._additional_output:
                return (cls_pred, box_pred, rpn_box, samples, matches,
                        raw_rpn_score, raw_rpn_box, anchors, top_feat)
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)

        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, dim=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.permute((0, 2, 1)).unsqueeze(3)
        scores = scores.permute((0, 2, 1)).unsqueeze(3)
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.permute((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(rpn_box, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(cls_ids, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        scores = _split(scores, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(box_pred, axis=0, num_outputs=self._max_batch, squeeze_axis=True)

        # per batch predict, nms, each class has topk outputs
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            # res (C, N, 6)
            res = torch.cat([cls_id, score, bbox], dim=-1)
            # res (C, self.nms_topk, 6)
            # res = box_nms_py(
            #     res, iou_threshold=self.nms_thresh, topk=self.nms_topk,
            #     score_index=1, coord_start=2)
            # res (C * self.nms_topk, 6)
            res = box_nms(res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=1e-4,
                          id_index=0, score_index=1, coord_start=2, force_suppress=True, sort=True)
            res = res.reshape((-1, 6))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = torch.stack(results, dim=0)
        ids = result.narrow(-1, 0, 1)
        scores = result.narrow(-1, 1, 1)
        bboxes = result.narrow(-1, 2, 4)
        if self._additional_output:
            return ids, scores, bboxes, feat
        return ids, scores, bboxes


def get_faster_rcnn(name, dataset, pretrained=False,
                    root=os.path.expanduser('~/.torch/models'), **kwargs):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = FasterRCNN(**kwargs)
    if pretrained:
        from model.model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_state_dict(torch.load(get_model_file(full_name, root=root)))
    return net


def faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from model.models_zoo.resnetv1b import resnet50_v1b
    from data.pascal_voc.detection_cv import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                **kwargs)
    features = list()
    top_features = list()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.append(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.append(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    features, top_features = nn.Sequential(*features), nn.Sequential(*top_features)
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_in_channel=1024, rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from model.models_zoo.resnetv1b import resnet50_v1b
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                **kwargs)
    features, top_features = list(), list()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.append(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.append(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    features, top_features = nn.Sequential(*features), nn.Sequential(*top_features)
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_in_channel=1024, rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from data.mscoco.detection_cv import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    features = FPNFeatureExpander(
        network='resnet50_v1b', outputs=[[4, 2, 5], [5, 3, 5], [6, 5, 5], [7, 2, 5]],
        channels=[64, 128, 256, 512], num_filters=[256, 256, 256, 256],
        use_1x1=True, use_upsample=True, use_elewadd=True,
        use_p6=True, use_bias=True, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    roi_size = 14
    box_features = nn.Sequential(
        nn.Linear(256 * roi_size ** 2, 1024), nn.ReLU(inplace=True),
        nn.Linear(1024, 1024), nn.ReLU(inplace=True),
    )

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(roi_size, roi_size),
        strides=(4, 8, 16, 32, 64), clip=4.42, channel=1024, rpn_in_channel=256, rpn_channel=1024,
        base_size=16, scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


if __name__ == '__main__':
    net = faster_rcnn_fpn_resnet50_v1b_coco()
    # print(net)
    net.eval()
    with torch.no_grad():
        a = torch.randn(1, 3, 150, 150)
        out = net(a)
        print(out)
