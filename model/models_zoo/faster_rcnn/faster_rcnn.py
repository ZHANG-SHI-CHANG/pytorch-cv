"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import warnings

from model.models_zoo.rcnn import RCNN
from model.models_zoo.faster_rcnn import RCNNTargetGenerator
from model.models_zoo.rpn import RPN


class FasterRCNN(RCNN):
    def __init__(self, features, top_features, classes, box_features=None,
                 short=600, max_size=1000, min_stage=4, max_stage=4, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
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
            **kwargs)
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
            channels=rpn_channel, strides=strides, base_size=base_size,
            scales=scales, ratios=ratios, alloc_size=alloc_size,
            clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
            train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
            test_post_nms=rpn_test_post_nms, min_size=rpn_min_size,
            multi_level=self.num_stages > 1)
        # self.sampler = RCNNTargetSampler(
        #     num_image=self._max_batch, num_proposal=rpn_train_post_nms,
        #     num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
        #     pos_ratio=pos_ratio, max_num_gt=max_num_gt)

    # @property
    # def target_generator(self):
    #     """Returns stored target generator
    #
    #     Returns
    #     -------
    #     mxnet.gluon.HybridBlock
    #         The RCNN target generator
    #
    #     """
    #     return list(self._target_generator)[0]
    #
    # def reset_class(self, classes, reuse_weights=None):
    #     """Reset class categories and class predictors.
    #
    #     Parameters
    #     ----------
    #     classes : iterable of str
    #         The new categories. ['apple', 'orange'] for example.
    #     reuse_weights : dict
    #         A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
    #         or a list of [name0, name1,...] if class names don't change.
    #         This allows the new predictor to reuse the
    #         previously trained weights specified.
    #
    #     Example
    #     -------
    #     >>> net = gluoncv.model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
    #     >>> # use direct name to name mapping to reuse weights
    #     >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
    #     >>> # or use interger mapping, person is the 14th category in VOC
    #     >>> net.reset_class(classes=['person'], reuse_weights={0:14})
    #     >>> # you can even mix them
    #     >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
    #     >>> # or use a list of string if class name don't change
    #     >>> net.reset_class(classes=['person'], reuse_weights=['person'])
    #
    #     """
    #     super(FasterRCNN, self).reset_class(classes, reuse_weights)
    #     self._target_generator = {RCNNTargetGenerator(self.num_class)}
    #
    # def _pyramid_roi_feats(self, F, features, rpn_rois, roi_size, strides, roi_mode='align',
    #                        eps=1e-6):
    #     """Assign rpn_rois to specific FPN layers according to its area
    #        and then perform `ROIPooling` or `ROIAlign` to generate final
    #        region proposals aggregated features.
    #     Parameters
    #     ----------
    #     features : list of mx.ndarray or mx.symbol
    #         Features extracted from FPN base network
    #     rpn_rois : mx.ndarray or mx.symbol
    #         (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
    #     roi_size : tuple
    #         The size of each roi with regard to ROI-Wise operation
    #         each region proposal will be roi_size spatial shape.
    #     strides : tuple e.g. [4, 8, 16, 32]
    #         Define the gap that ori image and feature map have
    #     roi_mode : str, default is align
    #         ROI pooling mode. Currently support 'pool' and 'align'.
    #     Returns
    #     -------
    #     Pooled roi features aggregated according to its roi_level
    #     """
    #     max_stage = self._max_stage
    #     if self._max_stage > 5:  # do not use p6 for RCNN
    #         max_stage = self._max_stage - 1
    #     _, x1, y1, x2, y2 = F.split(rpn_rois, axis=-1, num_outputs=5)
    #     h = y2 - y1 + 1
    #     w = x2 - x1 + 1
    #     roi_level = F.floor(4 + F.log2(F.sqrt(w * h) / 224.0 + eps))
    #     roi_level = F.squeeze(F.clip(roi_level, self._min_stage, max_stage))
    #     # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
    #     # roi_level_sorted_args = F.argsort(roi_level, is_ascend=True)
    #     # roi_level = F.sort(roi_level, is_ascend=True)
    #     # rpn_rois = F.take(rpn_rois, roi_level_sorted_args, axis=0)
    #     pooled_roi_feats = []
    #     for i, l in enumerate(range(self._min_stage, max_stage + 1)):
    #         # Pool features with all rois first, and then set invalid pooled features to zero,
    #         # at last ele-wise add together to aggregate all features.
    #         if roi_mode == 'pool':
    #             pooled_feature = F.ROIPooling(features[i], rpn_rois, roi_size, 1. / strides[i])
    #         elif roi_mode == 'align':
    #             pooled_feature = F.contrib.ROIAlign(features[i], rpn_rois, roi_size,
    #                                                 1. / strides[i],
    #                                                 sample_ratio=2)
    #         else:
    #             raise ValueError("Invalid roi mode: {}".format(roi_mode))
    #         pooled_feature = F.where(roi_level == l, pooled_feature, F.zeros_like(pooled_feature))
    #         pooled_roi_feats.append(pooled_feature)
    #     # Ele-wise add to aggregate all pooled features
    #     pooled_roi_feats = F.ElementWiseSum(*pooled_roi_feats)
    #     # Sort all pooled features by asceding order
    #     # [2,2,..,3,3,...,4,4,...,5,5,...]
    #     # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
    #     # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
    #     return pooled_roi_feats
    #
    # # pylint: disable=arguments-differ
    # def hybrid_forward(self, F, x, gt_box=None):
    #     """Forward Faster-RCNN network.
    #
    #     The behavior during training and inference is different.
    #
    #     Parameters
    #     ----------
    #     x : mxnet.nd.NDArray or mxnet.symbol
    #         The network input tensor.
    #     gt_box : type, only required during training
    #         The ground-truth bbox tensor with shape (1, N, 4).
    #
    #     Returns
    #     -------
    #     (ids, scores, bboxes)
    #         During inference, returns final class id, confidence scores, bounding
    #         boxes.
    #
    #     """
    #
    #     def _split(x, axis, num_outputs, squeeze_axis):
    #         x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
    #         if isinstance(x, list):
    #             return x
    #         else:
    #             return [x]
    #
    #     feat = self.features(x)
    #     if not isinstance(feat, (list, tuple)):
    #         feat = [feat]
    #
    #     # RPN proposals
    #     if autograd.is_training():
    #         rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = \
    #             self.rpn(F.zeros_like(x), *feat)
    #         rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_box)
    #     else:
    #         _, rpn_box = self.rpn(F.zeros_like(x), *feat)
    #
    #     # create batchid for roi
    #     num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
    #     with autograd.pause():
    #         # roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
    #         roi_batchid = F.arange(0, self._max_batch)
    #         roi_batchid = F.repeat(roi_batchid, num_roi)
    #         # remove batch dim because ROIPooling require 2d input
    #         rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)
    #         rpn_roi = F.stop_gradient(rpn_roi)
    #
    #     if self.num_stages > 1:
    #         # using FPN
    #         pooled_feat = self._pyramid_roi_feats(F, feat, rpn_roi, self._roi_size,
    #                                               self._strides, roi_mode=self._roi_mode)
    #     else:
    #         # ROI features
    #         if self._roi_mode == 'pool':
    #             pooled_feat = F.ROIPooling(feat[0], rpn_roi, self._roi_size, 1. / self._strides)
    #         elif self._roi_mode == 'align':
    #             pooled_feat = F.contrib.ROIAlign(feat[0], rpn_roi, self._roi_size,
    #                                              1. / self._strides, sample_ratio=2)
    #         else:
    #             raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
    #
    #     # RCNN prediction
    #     if self.top_features is not None:
    #         top_feat = self.top_features(pooled_feat)
    #     else:
    #         top_feat = pooled_feat
    #     if self.box_features is None:
    #         box_feat = F.contrib.AdaptiveAvgPooling2D(top_feat, output_size=1)
    #     else:
    #         box_feat = self.box_features(top_feat)
    #     cls_pred = self.class_predictor(box_feat)
    #     box_pred = self.box_predictor(box_feat)
    #     # cls_pred (B * N, C) -> (B, N, C)
    #     cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
    #     # box_pred (B * N, C * 4) -> (B, N, C, 4)
    #     box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_class, 4))
    #
    #     # no need to convert bounding boxes in training, just return
    #     if autograd.is_training():
    #         if self._additional_output:
    #             return (cls_pred, box_pred, rpn_box, samples, matches,
    #                     raw_rpn_score, raw_rpn_box, anchors, top_feat)
    #         return (cls_pred, box_pred, rpn_box, samples, matches,
    #                 raw_rpn_score, raw_rpn_box, anchors)
    #
    #     # cls_ids (B, N, C), scores (B, N, C)
    #     cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
    #     # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
    #     cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
    #     scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
    #     # box_pred (B, N, C, 4) -> (B, C, N, 4)
    #     box_pred = box_pred.transpose((0, 2, 1, 3))
    #
    #     # rpn_boxes (B, N, 4) -> B * (1, N, 4)
    #     rpn_boxes = _split(rpn_box, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
    #     # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
    #     cls_ids = _split(cls_ids, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
    #     scores = _split(scores, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
    #     # box_preds (B, C, N, 4) -> B * (C, N, 4)
    #     box_preds = _split(box_pred, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
    #
    #     # per batch predict, nms, each class has topk outputs
    #     results = []
    #     for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
    #         # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
    #         bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
    #         # res (C, N, 6)
    #         res = F.concat(*[cls_id, score, bbox], dim=-1)
    #         # res (C, self.nms_topk, 6)
    #         res = F.contrib.box_nms(
    #             res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
    #             id_index=0, score_index=1, coord_start=2, force_suppress=True)
    #         # res (C * self.nms_topk, 6)
    #         res = res.reshape((-3, 0))
    #         results.append(res)
    #
    #     # result B * (C * topk, 6) -> (B, C * topk, 6)
    #     result = F.stack(*results, axis=0)
    #     ids = F.slice_axis(result, axis=-1, begin=0, end=1)
    #     scores = F.slice_axis(result, axis=-1, begin=1, end=2)
    #     bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
    #     if self._additional_output:
    #         return ids, scores, bboxes, feat
    #     return ids, scores, bboxes
