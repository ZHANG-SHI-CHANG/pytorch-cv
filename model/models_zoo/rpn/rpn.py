"""Region Proposal Networks Definition."""
from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F

from model.models_zoo.rpn import RPNAnchorGenerator, RPNProposal
from model.ops import box_nms


class RPNHead(nn.Module):
    r"""Region Proposal Network Head.

    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    anchor_depth : int
        Each FPN stage have one scale and three ratios,
        we can compute anchor_depth = len(scale) \times len(ratios)
    """

    def __init__(self, in_channels, channels, anchor_depth, **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 1, 1), nn.ReLU(inplace=True))
        # use sigmoid instead of softmax, reduce channel numbers
        # Note : that is to say, if use softmax here,
        # then the self.score will anchor_depth*2 output channel
        self.score = nn.Conv2d(channels, anchor_depth, 1, 1, 0)
        self.loc = nn.Conv2d(channels, anchor_depth * 4, 1, 1, 0)

    # TODO: check stop_gradient
    def forward(self, x):
        """Forward RPN Head.

        This HybridBlock will generate predicted values for cls and box.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            Feature tensor. With (1, C, H, W) shape

        Returns
        -------
        (rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes)
            Returns predicted scores and regions.

        """
        # 3x3 conv with relu activation
        x = self.conv1(x)
        # (1, C, H, W)->(1, 9, H, W)->(1, H, W, 9)->(1, H*W*9, 1)
        raw_rpn_scores = self.score(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        # (1, H*W*9, 1)
        rpn_scores = torch.sigmoid(raw_rpn_scores.detach())
        # (1, C, H, W)->(1, 36, H, W)->(1, H, W, 36)->(1, H*W*9, 4)
        raw_rpn_boxes = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
        # (1, H*W*9, 1)
        rpn_boxes = raw_rpn_boxes.detach()
        # return raw predictions as well in training for bp
        return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes


class RPN(nn.Module):
    def __init__(self, in_channels, channels, strides, base_size, scales, ratios, alloc_size,
                 clip, nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size, multi_level=False, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._nms_thresh = nms_thresh
        self._multi_level = multi_level
        self._train_pre_nms = max(1, train_pre_nms)
        self._train_post_nms = max(1, train_post_nms)
        self._test_pre_nms = max(1, test_pre_nms)
        self._test_post_nms = max(1, test_post_nms)
        num_stages = len(scales)
        if self._multi_level:
            asz = alloc_size
            self.anchor_generator = list()
            for _, st, s in zip(range(num_stages), strides, scales):
                stage_anchor_generator = RPNAnchorGenerator(st, base_size, ratios, s, asz)
                self.anchor_generator.append(stage_anchor_generator)
                asz = max(asz[0] // 2, 16)
                asz = (asz, asz)  # For FPN, We use large anchor presets
            anchor_depth = self.anchor_generator[0].num_depth
            self.rpn_head = RPNHead(in_channels, channels, anchor_depth)
        else:
            self.anchor_generator = RPNAnchorGenerator(
                strides, base_size, ratios, scales, alloc_size)
            anchor_depth = self.anchor_generator.num_depth
            # not using RPNHead to keep backward compatibility with old models
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 1, 1), nn.ReLU(inplace=True))
            self.score = nn.Conv2d(channels, anchor_depth, 1, 1, 0)
            self.loc = nn.Conv2d(channels, anchor_depth * 4, 1, 1, 0)

        self.region_proposer = RPNProposal(clip, min_size, stds=(1., 1., 1., 1.))

    # pylint: disable=arguments-differ
    def forward(self, img, *x):
        if self.training:
            pre_nms = self._train_pre_nms
            post_nms = self._train_post_nms
        else:
            pre_nms = self._test_pre_nms
            post_nms = self._test_post_nms
        anchors = []
        rpn_pre_nms_proposals = []
        raw_rpn_scores = []
        raw_rpn_boxes = []
        if self._multi_level:
            # Generate anchors in [P2, P3, P4, P5, P6] order
            for i, feat in enumerate(x):
                ag = self.anchor_generator[i]
                anchor = ag(feat)
                rpn_score, rpn_box, raw_rpn_score, raw_rpn_box = \
                    self.rpn_head(feat)
                rpn_pre = self.region_proposer(anchor, rpn_score,
                                               rpn_box, img)
                anchors.append(anchor)
                rpn_pre_nms_proposals.append(rpn_pre)
                raw_rpn_scores.append(raw_rpn_score)
                raw_rpn_boxes.append(raw_rpn_box)
            rpn_pre_nms_proposals = torch.cat(rpn_pre_nms_proposals, dim=1)
            raw_rpn_scores = torch.cat(raw_rpn_scores, dim=1)
            raw_rpn_boxes = torch.cat(raw_rpn_boxes, dim=1)
        else:
            x = x[0]
            b = x.shape[0]
            anchors = self.anchor_generator(x)
            x = self.conv1(x)
            raw_rpn_scores = self.score(x).permute(0, 2, 3, 1).reshape((b, -1, 1))
            rpn_scores = torch.sigmoid(raw_rpn_scores.detach())
            raw_rpn_boxes = self.loc(x).permute(0, 2, 3, 1).reshape((b, -1, 4))
            rpn_boxes = raw_rpn_boxes.detach()
            rpn_pre_nms_proposals = self.region_proposer(
                anchors, rpn_scores, rpn_boxes, img)

        # Non-maximum suppression
        tmp = box_nms(rpn_pre_nms_proposals, overlap_thresh=self._nms_thresh, topk=pre_nms,
                      coord_start=1, score_index=0, id_index=-1, force_suppress=True, sort=True)

        # slice post_nms number of boxes
        result = tmp.narrow(1, 0, post_nms)
        rpn_scores = result.narrow(-1, 0, 1)
        rpn_boxes = result.narrow(-1, 1, result.shape[-1] - 1)

        if self.training:
            # return raw predictions as well in training for bp
            return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes, anchors
        return rpn_scores, rpn_boxes

    def get_test_post_nms(self):
        return self._test_post_nms
