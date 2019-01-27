"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import

import torch
from torch import nn

from model.module.bbox import BBoxCornerToCenter


class NormalizedBoxCenterDecoder(nn.Module):
    """Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).
    clip: float, default is None
        If given, bounding box target will be clipped to this value.

    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.),
                 convert_anchor=False, clip=None):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self._clip = clip
        if convert_anchor:
            self.corner_to_center = BBoxCornerToCenter(split=True)
        else:
            self.corner_to_center = None

    def forward(self, x, anchors):
        if self.corner_to_center is not None:
            a = self.corner_to_center(anchors)
        else:
            a = anchors.split(1, -1)
        p = torch.split(x, 1, -1)
        ox = (p[0] * self._stds[0] + self._means[0]) * a[2] + a[0]
        oy = (p[1] * self._stds[1] + self._means[1]) * a[3] + a[1]
        tw = torch.exp(p[2] * self._stds[2] + self._means[2])
        th = torch.exp(p[3] * self._stds[3] + self._means[3])
        if self._clip:
            tw.clamp_(0, self._clip)
            th.clamp_(0, self._clip)
        ow = (tw * a[2]) / 2
        oh = (th * a[3]) / 2
        return torch.cat([ox - ow, oy - oh, ox + ow, oy + oh], dim=-1)


class MultiPerClassDecoder(nn.Module):
    """Decode classification results.

    This decoder must work with `MultiClassEncoder` to reconstruct valid labels.
    The decoder expect results are after logits, e.g. Softmax.
    This version is different from
    :py:class:`gluoncv.nn.coder.MultiClassDecoder` with the following changes:

    For each position(anchor boxes), each foreground class can have their own
    results, rather than enforced to be the best one.
    For example, for a 5-class prediction with background(totaling 6 class), say
    (0.5, 0.1, 0.2, 0.1, 0.05, 0.05) as (bg, apple, orange, peach, grape, melon),
    `MultiClassDecoder` produce only one class id and score, that is  (orange-0.2).
    `MultiPerClassDecoder` produce 5 results individually:
    (apple-0.1, orange-0.2, peach-0.1, grape-0.05, melon-0.05).

    Parameters
    ----------
    num_class : int
        Number of classes including background.
    axis : int
        Axis of class-wise results.
    thresh : float
        Confidence threshold for the post-softmax scores.
        Scores less than `thresh` are marked with `0`, corresponding `cls_id` is
        marked with invalid class id `-1`.

    """

    def __init__(self, num_class, axis=-1, thresh=0.01):
        super(MultiPerClassDecoder, self).__init__()
        self._fg_class = num_class - 1
        self._axis = axis
        self._thresh = thresh

    def forward(self, x):
        scores = x.narrow(self._axis, 1, self._fg_class)  # b x N x fg_class
        template = torch.zeros_like(x.narrow(-1, 0, 1))
        cls_ids = []
        for i in range(self._fg_class):
            cls_ids.append(template + i)  # b x N x 1
        cls_id = torch.cat(cls_ids, dim=-1)  # b x N x fg_class
        mask = scores > self._thresh
        cls_id = torch.where(mask, cls_id, torch.ones_like(cls_id) * -1)
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        return cls_id, scores
