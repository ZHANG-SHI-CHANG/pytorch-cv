"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import

import torch
from torch import nn

from model.module.bbox import BBoxCornerToCenter


class NormalizedBoxCenterEncoder(nn.Module):
    """Encode bounding boxes training target with normalized center offsets.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self.corner_to_center = BBoxCornerToCenter(split=True)

    def forward(self, samples, matches, anchors, refs):
        """Not HybridBlock due to use of matches.shape

        Parameters
        ----------
        samples: (B, N) value +1 (positive), -1 (negative), 0 (ignore)
        matches: (B, N) value range [0, M)
        anchors: (B, N, 4) encoded in corner
        refs: (B, M, 4) encoded in corner

        Returns
        -------
        targets: (B, N, 4) transform anchors to refs picked according to matches
        masks: (B, N, 4) only positive anchors has targets

        """
        # TODO(zhreshold): batch_pick, take multiple elements?
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
        n, m = matches.shape[1], refs.shape[1]
        matches.clamp_(0).unsqueeze_(2)
        ref_boxes = refs.unsqueeze(1).repeat(1, n, 1, 1)
        # refs [B, N, M, 4] -> 4 * [B, N, M]
        ref_boxes = torch.split(ref_boxes, 1, -1)
        # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
        ref_boxes = torch.cat([torch.gather(ref_boxes[i].squeeze(-1), 2, matches) \
                               for i in range(4)], dim=2)
        # transform based on x, y, w, h
        # g [B, N, 4], a [B, N, 4] -> codecs [B, N, 4]
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (torch.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (torch.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        codecs = torch.cat([t0, t1, t2, t3], dim=2)
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> boolean
        temp = samples.unsqueeze(2).repeat(1, 1, 4) > 0.5
        # fill targets and masks [B, N, 4]
        targets = torch.where(temp, codecs, torch.zeros_like(codecs))
        masks = torch.where(temp, torch.ones_like(temp), torch.zeros_like(temp))
        return targets, masks


if __name__ == '__main__':
    pass


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


class MultiClassEncoder(nn.Module):
    """Encode classification training target given matching results.

    This encoder will assign training target of matched bounding boxes to
    ground-truth label + 1 and negative samples with label 0.
    Ignored samples will be assigned with `ignore_label`, whose default is -1.

    Parameters
    ----------
    ignore_label : float
        Assigned to un-matched samples, they are neither positive or negative during
        training, and should be excluded in loss function. Default is -1.

    """

    def __init__(self, ignore_label=-1):
        super(MultiClassEncoder, self).__init__()
        self._ignore_label = ignore_label

    def forward(self, samples, matches, refs):
        """HybridBlock, handle multi batch correctly

        Parameters
        ----------
        samples: (B, N), value +1 (positive), -1 (negative), 0 (ignore)
        matches: (B, N), value range [0, M)
        refs: (B, M), value range [0, num_fg_class), excluding background

        Returns
        -------
        targets: (B, N), value range [0, num_fg_class + 1), including background

        """
        # samples (B, N) (+1, -1, 0: ignore), matches (B, N) [0, M), refs (B, M)
        # reshape refs (B, M) -> (B, 1, M) -> (B, N, M)
        n, m = matches.shape[1], refs.shape[1]
        refs = refs.unsqueeze(1).repeat(1, n, 1)
        # ids (B, N, M) -> (B, N), value [0, M + 1), 0 reserved for background class
        target_ids = torch.gather(refs, 2, matches.clamp(0, m).unsqueeze(2)).squeeze(2) + 1
        # samples 0: set ignore samples to ignore_label
        targets = torch.where(samples > 0.5, target_ids, torch.ones_like(target_ids) * self._ignore_label)
        # samples -1: set negative samples to 0
        targets = torch.where(samples < -0.5, torch.zeros_like(targets), targets)
        return targets

# if __name__ == '__main__':
#     samples = torch.tensor([[1, 0, -1, 1], [1, 0, 0, -1]])
#     matches = torch.tensor([[1, 0, 2, 1], [2, 1, 1, 0]])
#     refs = torch.tensor([[2, 1, 0], [0, 1, 2]])
#     coder = MultiClassEncoder()
#     out = coder(samples, matches, refs)
#     print(out)
