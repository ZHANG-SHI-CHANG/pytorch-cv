from itertools import product

import torch
from torch import nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, in_channels, channels, mask_dim, use_last_relu=False):
        super(ProtoNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 1, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, 1, 1),
                                   nn.ReLU(inplace=True), nn.Conv2d(channels, mask_dim, 1, 1))
        self.use_last_relu = use_last_relu

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        if self.use_last_relu:
            x = F.relu(x)
        return x


# TODO: 1. make it more flexible, move priors to self.init
#       2. the parameter share is a little strange
class PredictHead(nn.Module):
    def __init__(self, in_channels, channels, parent=None, aspect_ratios=[[1]], scales=[1],
                 max_size=550, num_priors=3, num_classes=81, mask_dim=32, mask_branch=True,
                 use_pixel_scales=True, mask_activate=torch.tanh):
        super(PredictHead, self).__init__()
        self.parent = [parent]
        if parent is None:
            self.up_feature = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 1, 1), nn.ReLU(inplace=True))
            self.bbox_layer = nn.Conv2d(channels, num_priors * 4, 3, 1, 1)
            self.conf_layer = nn.Conv2d(channels, num_priors * num_classes, 3, 1, 1)
            self.mask_layer = nn.Conv2d(channels, num_priors * mask_dim, 3, 1, 1)

        self.mask_branch, self.mask_activate = mask_branch, mask_activate
        self.num_classes, self.mask_dim = num_classes, mask_dim
        self.aspect_ratios, self.scales = aspect_ratios, scales
        self.max_size, self.last_conv_size = max_size, None
        self.use_pixel_scales = use_pixel_scales

    def forward(self, x):
        src = self if self.parent[0] is None else self.parent[0]
        b, _, h, w = x.size()
        x = src.up_feature(x)
        bbox = src.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
        conf = src.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(b, -1, self.num_classes)
        if self.mask_branch:
            mask = src.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
            mask = self.mask_activate(mask)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)
        priors = self.make_priors(h, w).to(bbox.device)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        return preds

    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """

        if self.last_conv_size != (conv_w, conv_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for scale, ars in zip(self.scales, self.aspect_ratios):
                    for ar in ars:
                        if self.use_pixel_scales:
                            w = scale * ar / self.max_size
                            h = scale * ar / self.max_size
                        else:
                            w = scale * ar / conv_w
                            h = scale / ar / conv_h

                        prior_data += [x, y, w, h]

            self.priors = torch.Tensor(prior_data).view(-1, 4)
            self.last_conv_size = (conv_w, conv_h)

        return self.priors
