"""RPN proposals."""
from __future__ import absolute_import

import torch
from torch import nn

from model.module.bbox import BBoxCornerToCenter, BBoxClipToImage
from model.module.coder import NormalizedBoxCenterDecoder


class RPNProposal(nn.Module):
    def __init__(self, clip, min_size, stds):
        super(RPNProposal, self).__init__()
        self._box_to_center = BBoxCornerToCenter()
        self._box_decoder = NormalizedBoxCenterDecoder(stds=stds, clip=clip)
        self._clipper = BBoxClipToImage()
        # self._compute_area = BBoxArea()
        self._min_size = min_size

    # pylint: disable=arguments-differ
    def forward(self, anchor, score, bbox_pred, img):
        """
        Generate proposals. Limit to batch-size=1 in current implementation.
        """
        # restore bounding boxes
        roi = self._box_decoder(bbox_pred, self._box_to_center(anchor))

        # clip rois to image's boundary
        # roi = F.Custom(roi, img, op_type='bbox_clip_to_image')
        roi = self._clipper(roi, img)

        # remove bounding boxes that don't meet the min_size constraint
        # by setting them to (-1, -1, -1, -1)
        # width = roi.slice_axis(axis=-1, begin=2, end=3)
        # height = roi.slice_axis(axis=-1, begin=3, end=None)
        xmin, ymin, xmax, ymax = roi.split(1, dim=-1)
        width = xmax - xmin
        height = ymax - ymin
        # TODO:(zhreshold), there's im_ratio to handle here, but it requires
        # add' info, and we don't expect big difference
        invalid = (width < self._min_size) | (height < self._min_size)

        # # remove out of bound anchors
        # axmin, aymin, axmax, aymax = F.split(anchor, axis=-1, num_outputs=4)
        # # it's a bit tricky to get right/bottom boundary in hybridblock
        # wrange = F.arange(0, 2560).reshape((1, 1, 1, 2560)).slice_like(
        #    img, axes=(3)).max().reshape((1, 1, 1))
        # hrange = F.arange(0, 2560).reshape((1, 1, 2560, 1)).slice_like(
        #    img, axes=(2)).max().reshape((1, 1, 1))
        # invalid = (axmin < 0) + (aymin < 0) + F.broadcast_greater(axmax, wrange) + \
        #    F.broadcast_greater(aymax, hrange)
        # avoid invalid anchors suppress anchors with 0 confidence
        score = torch.where(invalid, torch.ones_like(invalid, dtype=score.dtype) * -1, score)
        invalid = invalid.repeat(1, 1, 4)
        roi = torch.where(invalid, torch.ones_like(invalid, dtype=roi.dtype) * -1, roi)

        pre = torch.cat([score, roi], dim=-1)
        return pre
