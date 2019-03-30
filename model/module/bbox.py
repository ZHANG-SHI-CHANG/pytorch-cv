"""Bounding boxes operators"""
from __future__ import absolute_import

import torch
from torch import nn


class BBoxCornerToCenter(nn.Module):
    """Convert corner boxes to center boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 tensor if split is False, or 4 BxNx1 tensor if split is True
    """

    def __init__(self, axis=-1, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def forward(self, x):
        xmin, ymin, xmax, ymax = torch.split(x, 1, dim=self._axis)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not self._split:
            return torch.cat([x, y, width, height], dim=self._axis)
        else:
            return x, y, width, height


class BBoxCenterToCorner(nn.Module):
    """Convert center boxes to corner boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 tensor if split is False, or 4 BxNx1 tensor if split is True.
    """

    def __init__(self, axis=-1, split=False):
        super(BBoxCenterToCorner, self).__init__()
        self._split = split
        self._axis = axis

    def forward(self, x):
        x, y, w, h = torch.split(x, 1, dim=self._axis)
        hw = w / 2
        hh = h / 2
        xmin = x - hw
        ymin = y - hh
        xmax = x + hw
        ymax = y + hh
        if not self._split:
            return torch.cat([xmin, ymin, xmax, ymax], dim=self._axis)
        else:
            return xmin, ymin, xmax, ymax


class BBoxSplit(nn.Module):
    """Split bounding boxes into 4 columns.

    Parameters
    ----------
    axis : int, default is -1
        On which axis to split the bounding box. Default is -1(the last dimension).
    squeeze_axis : boolean, default is `False`
        If true, Removes the axis with length 1 from the shapes of the output arrays.
        **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only
        along the `axis` which it is split.
        Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.

    """

    def __init__(self, axis, squeeze_axis=False, **kwargs):
        super(BBoxSplit, self).__init__(**kwargs)
        self._axis = axis
        self._squeeze_axis = squeeze_axis

    def forward(self, x):
        if self._squeeze_axis:
            return tuple(x.squeeze_(self._axis) for x in torch.split(x, 1, dim=self._axis))
        else:
            return torch.split(x, 1, dim=self._axis)


class BBoxBatchIOU(nn.Module):
    """Batch Bounding Box IOU.

    Parameters
    ----------
    axis : int
        On which axis is the length-4 bounding box dimension.
    fmt : str
        BBox encoding format, can be 'corner' or 'center'.
        'corner': (xmin, ymin, xmax, ymax)
        'center': (center_x, center_y, width, height)
    offset : float, default is 0
        Offset is used if +1 is desired for computing width and height, otherwise use 0.
    eps : float, default is 1e-15
        Very small number to avoid division by 0.

    """

    def __init__(self, axis=-1, fmt='corner', offset=0, eps=1e-15, **kwargs):
        super(BBoxBatchIOU, self).__init__(**kwargs)
        self._offset = offset
        self._eps = eps
        if fmt.lower() == 'center':
            self._pre = BBoxCenterToCorner(split=True)
        elif fmt.lower() == 'corner':
            self._pre = BBoxSplit(axis=axis, squeeze_axis=True)
        else:
            raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))

    def forward(self, a, b):
        """Compute IOU for each batch

        Parameters
        ----------
        a : tensor
            (B, N, 4) first input.
        b : tensor
            (B, M, 4) second input.

        Returns
        -------
        tensor
            (B, N, M) array of IOUs.

        """
        al, at, ar, ab = self._pre(a)
        bl, bt, br, bb = self._pre(b)

        b, n, m = a.shape[0], a.shape[1], b.shape[1]
        # (B, N, M)
        left = torch.max(al.unsqueeze(2).expand(b, n, m), bl.unsqueeze(1).expand(b, n, m))
        right = torch.min(ar.unsqueeze(2).expand(b, n, m), br.unsqueeze(1).expand(b, n, m))
        top = torch.max(at.unsqueeze(2).expand(b, n, m), bt.unsqueeze(1).expand(b, n, m))
        bot = torch.min(ab.unsqueeze(2).expand(b, n, m), bb.unsqueeze(1).expand(b, n, m))

        # clip with (0, float16.max)
        iw = torch.clamp(right - left + self._offset, min=0, max=6.55040e+04)
        ih = torch.clamp(bot - top + self._offset, min=0, max=6.55040e+04)
        i = iw * ih

        # areas
        area_a = ((ar - al + self._offset) * (ab - at + self._offset)).unsqueeze(-1)
        area_b = ((br - bl + self._offset) * (bb - bt + self._offset)).unsqueeze(-2)
        union = area_a + area_b - i

        return i / (union + self._eps)

def box_iou(lhs, rhs):
    n, m = a.shape[0], a.shape[1], b.shape[1]


if __name__ == '__main__':
    a = torch.FloatTensor([[1.0, 1.0, 2.0, 2.0], [1.5, 1.5, 3.0, 3.0]]).view((2, 1, 4))
    b = torch.FloatTensor([[1.2, 1.2, 2.2, 2.3], [1.6, 1.6, 3.5, 3.5],
                           [1.4, 1.4, 2.2, 2.3], [1.3, 1.6, 3.5, 3.5]]).reshape((2, 2, 4))
    op = BBoxBatchIOU()
    out = op(a, b)
    print(out)
