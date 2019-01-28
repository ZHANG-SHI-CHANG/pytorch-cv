"""Anchor box generator for SSD detector."""
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn


class SSDAnchorGenerator(nn.Module):
    """Bounding box anchor generator for Single-shot Object Detection.

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).

    """

    def __init__(self, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(SSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        anchors = nn.Parameter(torch.from_numpy(anchors), requires_grad=False)
        self.register_parameter('anchor', anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                anchors.append([cx, cy, sizes[0], sizes[0]])
                anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1).astype(np.float32)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    def forward(self, x):
        a = self.anchor.narrow(2, 0, x.shape[2]).narrow(3, 0, x.shape[3])
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(1, dim=-1)
            H, W = self._im_size
            a = torch.cat([cx.clamp(0, W), cy.clamp(0, H), cw.clamp(0, W), ch.clamp(0, H)], dim=-1)
        return a.reshape((1, -1, 4))


if __name__ == '__main__':
    anchor = SSDAnchorGenerator(clip=True, im_size=(300, 300), sizes=(30, 60), ratios=[1, 2, 0.5], step=8,
                                alloc_size=(128, 128))
    a = torch.randn((1, 512, 64, 64))
    print(anchor(a))
