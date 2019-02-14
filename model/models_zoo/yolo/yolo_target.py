# """Target generators for YOLOs."""
# from __future__ import absolute_import
# from __future__ import division
#
# import numpy as np
#
# import torch
# from torch import nn
# from model.module.bbox import BBoxBatchIOU
#
#
# class YOLOV3DynamicTargetGeneratorSimple(nn.Module):
#     """YOLOV3 target generator that requires network predictions.
#     `Dynamic` indicate that the targets generated depend on current network.
#     `Simple` indicate that it only support `pos_iou_thresh` >= 1.0,
#     otherwise it's a lot more complicated and slower.
#     (box regression targets and class targets are not necessary when `pos_iou_thresh` >= 1.0)
#
#     Parameters
#     ----------
#     num_class : int
#         Number of foreground classes.
#     ignore_iou_thresh : float
#         Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
#         penalized of objectness score.
#
#     """
#
#     def __init__(self, num_class, ignore_iou_thresh, **kwargs):
#         super(YOLOV3DynamicTargetGeneratorSimple, self).__init__(**kwargs)
#         self._num_class = num_class
#         self._ignore_iou_thresh = ignore_iou_thresh
#         self._batch_iou = BBoxBatchIOU()
#
#     def forward(self, box_preds, gt_boxes):
#         """
#         Parameters
#         ----------
#         box_preds : tensor
#             Predicted bounding boxes.
#         gt_boxes : tensor
#             Ground-truth bounding boxes.
#
#         Returns
#         -------
#         (tuple of) tensor
#             objectness: 0 for negative, 1 for positive, -1 for ignore.
#             center_targets: regression target for center x and y.
#             scale_targets: regression target for scale x and y.
#             weights: element-wise gradient weights for center_targets and scale_targets.
#             class_targets: a one-hot vector for classification.
#
#         """
#         with autograd.pause():
#             box_preds = box_preds.view((0, -1, 4))
#             objness_t = F.zeros_like(box_preds.slice_axis(axis=-1, begin=0, end=1))
#             center_t = F.zeros_like(box_preds.slice_axis(axis=-1, begin=0, end=2))
#             scale_t = F.zeros_like(box_preds.slice_axis(axis=-1, begin=0, end=2))
#             weight_t = F.zeros_like(box_preds.slice_axis(axis=-1, begin=0, end=2))
#             class_t = F.ones_like(objness_t.tile(reps=(self._num_class))) * -1
#             batch_ious = self._batch_iou(box_preds, gt_boxes)  # (B, N, M)
#             ious_max = batch_ious.max(axis=-1, keepdims=True)  # (B, N, 1)
#             objness_t = (ious_max > self._ignore_iou_thresh) * -1  # use -1 for ignored
#         return objness_t, center_t, scale_t, weight_t, class_t
#
#
# class YOLOV3TargetMerger(nn.Module):
#     """YOLOV3 target merger that merges the prefetched targets and dynamic targets.
#
#     Parameters
#     ----------
#     num_class : int
#         Number of foreground classes.
#     ignore_iou_thresh : float
#         Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
#         penalized of objectness score.
#
#     """
#
#     def __init__(self, num_class, ignore_iou_thresh, **kwargs):
#         super(YOLOV3TargetMerger, self).__init__(**kwargs)
#         self._num_class = num_class
#         self._dynamic_target = YOLOV3DynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
#         self._label_smooth = False
#
#     def hybrid_forward(self, F, box_preds, gt_boxes, obj_t, centers_t, scales_t, weights_t, clas_t):
#         """Short summary.
#
#         Parameters
#         ----------
#         F : mxnet.nd or mxnet.sym
#             `F` is mxnet.sym if hybridized or mxnet.nd if not.
#         box_preds : mxnet.nd.NDArray
#             Predicted bounding boxes.
#         gt_boxes : mxnet.nd.NDArray
#             Ground-truth bounding boxes.
#         obj_t : mxnet.nd.NDArray
#             Prefetched Objectness targets.
#         centers_t : mxnet.nd.NDArray
#             Prefetched regression target for center x and y.
#         scales_t : mxnet.nd.NDArray
#             Prefetched regression target for scale x and y.
#         weights_t : mxnet.nd.NDArray
#             Prefetched element-wise gradient weights for center_targets and scale_targets.
#         clas_t : mxnet.nd.NDArray
#             Prefetched one-hot vector for classification.
#
#         Returns
#         -------
#         (tuple of) mxnet.nd.NDArray
#             objectness: 0 for negative, 1 for positive, -1 for ignore.
#             center_targets: regression target for center x and y.
#             scale_targets: regression target for scale x and y.
#             weights: element-wise gradient weights for center_targets and scale_targets.
#             class_targets: a one-hot vector for classification.
#
#         """
#         with autograd.pause():
#             dynamic_t = self._dynamic_target(box_preds, gt_boxes)
#             # use fixed target to override dynamic targets
#             obj, centers, scales, weights, clas = zip(
#                 dynamic_t, [obj_t, centers_t, scales_t, weights_t, clas_t])
#             mask = obj[1] > 0
#             objectness = F.where(mask, obj[1], obj[0])
#             mask2 = mask.tile(reps=(2,))
#             center_targets = F.where(mask2, centers[1], centers[0])
#             scale_targets = F.where(mask2, scales[1], scales[0])
#             weights = F.where(mask2, weights[1], weights[0])
#             mask3 = mask.tile(reps=(self._num_class,))
#             class_targets = F.where(mask3, clas[1], clas[0])
#             smooth_weight = 1. / self._num_class
#             if self._label_smooth:
#                 smooth_weight = 1. / self._num_class
#                 class_targets = F.where(
#                     class_targets > 0.5, class_targets - smooth_weight, class_targets)
#                 class_targets = F.where(
#                     class_targets < -0.5, class_targets, F.ones_like(class_targets) * smooth_weight)
#             class_mask = mask.tile(reps=(self._num_class,)) * (class_targets >= 0)
#             return [F.stop_gradient(x) for x in [objectness, center_targets, scale_targets,
#                                                  weights, class_targets, class_mask]]