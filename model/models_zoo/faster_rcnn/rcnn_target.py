"""RCNN Target Generator."""
from __future__ import absolute_import

import torch
from torch import nn

from model.ops import bbox_iou
from model.module.coder import MultiClassEncoder, NormalizedPerClassBoxCenterEncoder


class RCNNTargetGenerator(nn.Module):
    """RCNN target encoder to generate matching target and regression target values.

    Parameters
    ----------
    num_class : int
        Number of total number of positive classes.
    means : iterable of float, default is (0., 0., 0., 0.)
        Mean values to be subtracted from regression targets.
    stds : iterable of float, default is (.1, .1, .2, .2)
        Standard deviations to be divided from regression targets.

    """

    def __init__(self, num_class, means=(0., 0., 0., 0.), stds=(.1, .1, .2, .2)):
        super(RCNNTargetGenerator, self).__init__()
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedPerClassBoxCenterEncoder(
            num_class=num_class, means=means, stds=stds)

    # pylint: disable=arguments-differ
    def forward(self, roi, samples, matches, gt_label, gt_box):
        """Components can handle batch images

        Parameters
        ----------
        roi: (B, N, 4), input proposals
        samples: (B, N), value +1: positive / -1: negative.
        matches: (B, N), value [0, M), index to gt_label and gt_box.
        gt_label: (B, M), value [0, num_class), excluding background class.
        gt_box: (B, M, 4), input ground truth box corner coordinates.

        Returns
        -------
        cls_target: (B, N), value [0, num_class + 1), including background.
        box_target: (B, N, C, 4), only foreground class has nonzero target.
        box_weight: (B, N, C, 4), only foreground class has nonzero weight.

        """
        # cls_target (B, N)
        cls_target = self._cls_encoder(samples, matches, gt_label)
        # box_target, box_weight (C, B, N, 4)
        box_target, box_mask = self._box_encoder(
            samples, matches, roi, gt_label, gt_box)
        # modify shapes to match predictions
        # box (C, B, N, 4) -> (B, N, C, 4)
        box_target = box_target.transpose((1, 2, 0, 3))
        box_mask = box_mask.transpose((1, 2, 0, 3))
        return cls_target, box_target, box_mask


class RCNNTargetSampler(nn.Module):
    """A sampler to choose positive/negative samples from RCNN Proposals

    Parameters
    ----------
    num_image: int
        Number of input images.
    num_proposal: int
        Number of input proposals.
    num_sample : int
        Number of samples for RCNN targets.
    pos_iou_thresh : float
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
        Proposal whose IOU smaller than ``pos_iou_thresh`` is regarded as negative samples.
    pos_ratio : float
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.

    """

    def __init__(self, num_image, num_proposal, num_sample, pos_iou_thresh, pos_ratio, max_num_gt):
        super(RCNNTargetSampler, self).__init__()
        self._num_image = num_image
        self._num_proposal = num_proposal
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh
        self._max_num_gt = max_num_gt

    # pylint: disable=arguments-differ
    def forward(self, rois, scores, gt_boxes):
        """Handle B=self._num_image by a for loop.

        Parameters
        ----------
        rois: (B, self._num_input, 4) encoded in (x1, y1, x2, y2).
        scores: (B, self._num_input, 1), value range [0, 1] with ignore value -1.
        gt_boxes: (B, M, 4) encoded in (x1, y1, x2, y2), invalid box should have area of 0.

        Returns
        -------
        rois: (B, self._num_sample, 4), randomly drawn from proposals
        samples: (B, self._num_sample), value +1: positive / 0: ignore / -1: negative.
        matches: (B, self._num_sample), value between [0, M)

        """
        # collect results into list
        new_rois = []
        new_samples = []
        new_matches = []
        for i in range(self._num_image):
            roi = rois[i]
            score = scores[i]
            gt_box = gt_boxes[i]
            gt_score = torch.ones_like(torch.sum(gt_box, dim=-1, keepdim=True))

            # concat rpn roi with ground truth
            all_roi = torch.cat([roi, gt_box], dim=0)
            all_score = torch.cat([score, gt_score], dim=0).squeeze(-1)
            # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # cannot do batch op, will get (B, N, B, M) ious
            ious = bbox_iou(all_roi, gt_box, fmt='corner')
            # match to argmax iou
            ious_max, ious_argmax = ious.max(axis=-1)
            # init with 2, which are neg samples
            mask = torch.ones_like(ious_max) * 2
            # mark all ignore to 0
            mask = torch.where(all_score < 0, torch.zeros_like(mask), mask)
            # mark positive samples with 3
            pos_mask = ious_max >= self._pos_iou_thresh
            mask = torch.where(pos_mask, torch.ones_like(mask) * 3, mask)

            # shuffle mask
            rand = torch.zeros(self._num_proposal + self._max_num_gt, ).uniform(0, 1)
            rand = torch.slice_like(rand, ious_argmax)
            index = torch.argsort(rand)
            mask = mask[index]
            ious_argmax = ious_argmax[index]

            # sample pos samples
            order = torch.argsort(mask, descending=True)
            topk = order.narrow(0, 0, self._max_pos)
            topk_indices = index[topk]
            topk_samples = mask[topk]
            topk_matches = ious_argmax[topk]
            # reset output: 3 pos 2 neg 0 ignore -> 1 pos -1 neg 0 ignore
            topk_samples = torch.where(topk_samples == 3, torch.ones_like(topk_samples), topk_samples)
            topk_samples = torch.where(topk_samples == 2, torch.ones_like(topk_samples) * -1, topk_samples)

            # sample neg samples
            index = index[self._max_pos:]
            mask = mask[self._max_pos:]
            ious_argmax = ious_argmax[self._max_pos:]
            # change mask: 4 neg 3 pos 0 ignore
            mask = torch.where(mask == 2, torch.ones_like(mask) * 4, mask)
            order = torch.argsort(mask, descending=True)
            num_neg = self._num_sample - self._max_pos
            bottomk = order.narrow(0, 0, num_neg)
            bottomk_indices = index[bottomk]
            bottomk_samples = mask[bottomk]
            bottomk_matches = ious_argmax[bottomk]
            # reset output: 4 neg 3 pos 0 ignore -> 1 pos -1 neg 0 ignore
            bottomk_samples = torch.where(bottomk_samples == 3, torch.ones_like(bottomk_samples),
                                          bottomk_samples)
            bottomk_samples = torch.where(bottomk_samples == 4, torch.ones_like(bottomk_samples) * -1,
                                          bottomk_samples)

            # output
            indices = torch.cat([topk_indices, bottomk_indices], dim=0)
            samples = torch.cat([topk_samples, bottomk_samples], dim=0)
            matches = torch.cat([topk_matches, bottomk_matches], dim=0)

            new_rois.append(all_roi[indices])
            new_samples.append(samples)
            new_matches.append(matches)
        # stack all samples together
        new_rois = torch.stack(*new_rois, axis=0)
        new_samples = torch.stack(*new_samples, axis=0)
        new_matches = torch.stack(*new_matches, axis=0)
        return new_rois, new_samples, new_matches
