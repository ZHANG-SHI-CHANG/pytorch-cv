"""Target generators for YOLOs."""
from __future__ import absolute_import
from __future__ import division

import math
import numpy as np
import torch
from torch import nn
from model.module.bbox import BBoxBatchIOU, BBoxCornerToCenter, BBoxCenterToCorner


class YOLOV3PrefetchTargetGenerator(nn.Module):
    """YOLO V3 prefetch target generator.
    The target generated by this instance is invariant to network predictions.
    Therefore it is usually used in DataLoader transform function to reduce the load on GPUs.

    Parameters
    ----------
    num_class : int
        Number of foreground classes.

    """

    def __init__(self, num_class, **kwargs):
        super(YOLOV3PrefetchTargetGenerator, self).__init__(**kwargs)
        self._num_class = num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)
        self.bbox_iou = BBoxBatchIOU()

    def forward(self, h, w, xs, anchors, offsets, gt_boxes, gt_ids, gt_mixratio=None):
        """Generating training targets that do not require network predictions.

        Parameters
        ----------
        img : tensor
            Original image tensor.
        xs : list of tensor
            List of feature maps.
        anchors : list of tensor
            YOLO3 anchors.
        offsets : list of tensor
            Pre-generated x and y offsets for YOLO3.
        gt_boxes : tensor
            Ground-truth boxes.
        gt_ids : tensor
            Ground-truth IDs.
        gt_mixratio : tensor, optional
            Mixup ratio from 0 to 1.

        Returns
        -------
        (tuple of) tensor
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        assert isinstance(anchors, (list, tuple))
        all_anchors = torch.cat([a.reshape(-1, 2) for a in anchors], dim=0)
        assert isinstance(offsets, (list, tuple))
        all_offsets = torch.cat([o.reshape(-1, 2) for o in offsets], dim=0)
        num_anchors = torch.cumsum(torch.tensor([a.numel() // 2 for a in anchors]), 0)
        num_offsets = torch.cumsum(torch.tensor([o.numel() // 2 for o in offsets]), 0)
        _offsets = [0] + num_offsets.tolist()
        assert isinstance(xs, (list, tuple))
        assert len(xs) == len(anchors) == len(offsets)

        # orig image size
        orig_height = h
        orig_width = w

        # outputs
        shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
            (-1, 1, 2)).unsqueeze(0).repeat(gt_ids.shape[0], 1, 1, 1)
        center_targets = torch.zeros_like(shape_like)
        scale_targets = torch.zeros_like(center_targets)
        weights = torch.zeros_like(center_targets)
        objectness = torch.zeros_like(weights.split(1, -1)[0])
        class_targets = -1 * torch.ones_like(objectness).repeat(1, 1, 1, self._num_class)

        # for each ground-truth, find the best matching anchor within the particular grid
        # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
        # then only the anchor in (3, 4) is going to be matched
        gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
        shift_gt_boxes = torch.cat([-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth], dim=-1)
        anchor_boxes = torch.cat([0 * all_anchors, all_anchors], dim=-1)  # zero center anchors
        shift_anchor_boxes = self.bbox2corner(anchor_boxes).unsqueeze(0).repeat(shift_gt_boxes.shape[0], 1, 1)
        ious = self.bbox_iou(shift_anchor_boxes, shift_gt_boxes)
        # TODO: rewrite it
        # real value is required to process, convert to Numpy
        matches = ious.argmax(1).numpy()  # (B, M)
        valid_gts = (gt_boxes >= 0).numpy().prod(axis=-1)  # (B, M)
        np_gtx, np_gty, np_gtw, np_gth = [x.numpy() for x in [gtx, gty, gtw, gth]]
        np_anchors = all_anchors.numpy()
        np_gt_ids = gt_ids.numpy()
        np_gt_mixratios = gt_mixratio.numpy() if gt_mixratio is not None else None
        # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
        # should not be a problem right now. Switch to better solution is needed.
        for b in range(matches.shape[0]):
            for m in range(matches.shape[1]):
                if valid_gts[b, m] < 1:
                    break
                match = int(matches[b, m])
                nlayer = np.nonzero(num_anchors > match)[0][0]
                height = xs[nlayer].shape[2]
                width = xs[nlayer].shape[3]
                gtx, gty, gtw, gth = (np_gtx[b, m, 0], np_gty[b, m, 0],
                                      np_gtw[b, m, 0], np_gth[b, m, 0])
                # compute the location of the gt centers
                loc_x = int(gtx / orig_width * width)
                loc_y = int(gty / orig_height * height)
                # write back to targets
                index = _offsets[nlayer] + loc_y * width + loc_x
                center_targets[b, index, match, 0] = gtx / orig_width * width - loc_x  # tx
                center_targets[b, index, match, 1] = gty / orig_height * height - loc_y  # ty
                scale_targets[b, index, match, 0] = math.log(max(gtw, 1) / np_anchors[match, 0])
                scale_targets[b, index, match, 1] = math.log(max(gth, 1) / np_anchors[match, 1])
                weights[b, index, match, :] = 2.0 - gtw * gth / orig_width / orig_height
                objectness[b, index, match, 0] = (
                    np_gt_mixratios[b, m, 0].item() if np_gt_mixratios is not None else 1)
                class_targets[b, index, match, :] = 0
                class_targets[b, index, match, int(np_gt_ids[b, m, 0])] = 1
        # since some stages won't see partial anchors, so we have to slice the correct targets
        objectness = self._slice(objectness, num_anchors, num_offsets)
        center_targets = self._slice(center_targets, num_anchors, num_offsets)
        scale_targets = self._slice(scale_targets, num_anchors, num_offsets)
        weights = self._slice(weights, num_anchors, num_offsets)
        class_targets = self._slice(class_targets, num_anchors, num_offsets)
        return objectness, center_targets, scale_targets, weights, class_targets

    def _slice(self, x, num_anchors, num_offsets):
        """since some stages won't see partial anchors, so we have to slice the correct targets"""
        # x with shape (B, N, A, 1 or 2)
        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i + 1], anchors[i]:anchors[i + 1], :]
            ret.append(y.reshape((y.shape[0], -1, y.shape[-1])))
        return torch.cat(ret, dim=1)


class YOLOV3DynamicTargetGeneratorSimple(nn.Module):
    """YOLOV3 target generator that requires network predictions.
    `Dynamic` indicate that the targets generated depend on current network.
    `Simple` indicate that it only support `pos_iou_thresh` >= 1.0,
    otherwise it's a lot more complicated and slower.
    (box regression targets and class targets are not necessary when `pos_iou_thresh` >= 1.0)

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.

    """

    def __init__(self, num_class, ignore_iou_thresh, **kwargs):
        super(YOLOV3DynamicTargetGeneratorSimple, self).__init__(**kwargs)
        self._num_class = num_class
        self._ignore_iou_thresh = ignore_iou_thresh
        self._batch_iou = BBoxBatchIOU()

    def forward(self, box_preds, gt_boxes):
        """
        Parameters
        ----------
        box_preds : tensor
            Predicted bounding boxes.
        gt_boxes : tensor
            Ground-truth bounding boxes.

        Returns
        -------
        (tuple of) tensor
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        b = box_preds.shape[0]
        box_preds = box_preds.view((b, -1, 4))
        objness_t = torch.zeros_like(box_preds.narrow(-1, 0, 1))
        center_t = torch.zeros_like(box_preds.narrow(-1, 0, 2))
        scale_t = torch.zeros_like(box_preds.narrow(-1, 0, 2))
        weight_t = torch.zeros_like(box_preds.narrow(-1, 0, 2))
        class_t = torch.ones_like(objness_t.repeat(1, 1, self._num_class)) * -1
        batch_ious = self._batch_iou(box_preds, gt_boxes)  # (B, N, M)
        ious_max, _ = batch_ious.max(dim=-1, keepdim=True)  # (B, N, 1)
        objness_t = (ious_max > self._ignore_iou_thresh).type(box_preds.dtype) * -1  # use -1 for ignored
        return objness_t, center_t, scale_t, weight_t, class_t


class YOLOV3TargetMerger(nn.Module):
    """YOLOV3 target merger that merges the prefetched targets and dynamic targets.

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.

    """

    def __init__(self, num_class, ignore_iou_thresh, **kwargs):
        super(YOLOV3TargetMerger, self).__init__(**kwargs)
        self._num_class = num_class
        self._dynamic_target = YOLOV3DynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
        self._label_smooth = False

    def forward(self, box_preds, gt_boxes, obj_t, centers_t, scales_t, weights_t, clas_t):
        """Short summary.

        Parameters
        ----------
        box_preds : tensor
            Predicted bounding boxes.
        gt_boxes : tensor
            Ground-truth bounding boxes.
        obj_t : tensor
            Prefetched Objectness targets.
        centers_t : tensor
            Prefetched regression target for center x and y.
        scales_t : tensor
            Prefetched regression target for scale x and y.
        weights_t : tensor
            Prefetched element-wise gradient weights for center_targets and scale_targets.
        clas_t : tensor
            Prefetched one-hot vector for classification.

        Returns
        -------
        (tuple of) tensor
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        dynamic_t = self._dynamic_target(box_preds, gt_boxes)
        # use fixed target to override dynamic targets
        obj, centers, scales, weights, clas = zip(dynamic_t, [obj_t, centers_t, scales_t, weights_t, clas_t])
        mask = (obj[1] > 0)
        objectness = torch.where(mask, obj[1], obj[0])
        mask2 = mask.repeat(1, 1, 2)
        center_targets = torch.where(mask2, centers[1], centers[0])
        scale_targets = torch.where(mask2, scales[1], scales[0])
        weights = torch.where(mask2, weights[1], weights[0])
        mask3 = mask.repeat(1, 1, self._num_class)
        class_targets = torch.where(mask3, clas[1], clas[0])
        if self._label_smooth:
            smooth_weight = 1. / self._num_class
            class_targets = torch.where(
                class_targets > 0.5, class_targets - smooth_weight, class_targets)
            class_targets = torch.where(
                class_targets < -0.5, class_targets, torch.ones_like(class_targets) * smooth_weight)
        class_mask = (mask3 * (class_targets >= 0)).type(weights.dtype)  # TODO: may need to change type
        return [objectness, center_targets, scale_targets,
                weights, class_targets, class_mask]


if __name__ == '__main__':
    import numpy as np

    np.random.seed(10)
    box_preds = np.random.random(size=(1, 10, 4))
    gt_boxes = np.random.random(size=(1, 2, 4))
    obj_t = np.random.random(size=(1, 10, 1))
    centers_t = np.random.random(size=(1, 10, 2))
    scales_t = np.random.random(size=(1, 10, 2))
    weights_t = np.random.random(size=(1, 10, 2))
    clas_t = np.random.random(size=(1, 10, 20))

    op = YOLOV3TargetMerger(20, 0.2)

    box_preds = torch.from_numpy(box_preds)
    gt_boxes = torch.from_numpy(gt_boxes)
    obj_t = torch.from_numpy(obj_t)
    centers_t = torch.from_numpy(centers_t)
    scales_t = torch.from_numpy(scales_t)
    weights_t = torch.from_numpy(weights_t)
    clas_t = torch.from_numpy(clas_t)

    out = op(box_preds, gt_boxes, obj_t, centers_t, scales_t, weights_t, clas_t)
    print(out)
