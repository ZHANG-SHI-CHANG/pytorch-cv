from torch import nn

from model.module.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from model.module.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from model.module.sampler import NaiveSampler
from model.module.bbox import BBoxCenterToCorner
from utils.bbox_pt import bbox_iou


class SSDTargetGenerator(nn.Module):
    """Training targets generator for Single-shot Object Detection.

    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """

    def __init__(self, iou_thresh=0.5,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher(
            [BipartiteMatcher(share_max=False), MaximumMatcher(iou_thresh)])
        self._sampler = NaiveSampler()
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, anchors, gt_boxes, gt_ids):
        """Generate training targets."""
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = bbox_iou(anchors, gt_boxes)
        matches = self._matcher(ious).unsqueeze(0)
        samples = self._sampler(matches)
        gt_boxes.unsqueeze_(0), gt_ids.unsqueeze_(0)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks
