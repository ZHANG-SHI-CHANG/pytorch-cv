"""Calculate Intersection-Over-Union(IOU) of two bounding boxes."""
from __future__ import division

import math
import torch

import numpy as np


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : torch.Tensor
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : torch.Tensor
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        An tensor with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")
    n, m = bbox_a.shape[0], bbox_b.shape[0]
    tl = torch.max(bbox_a[:, :2].unsqueeze(1).expand(n, m, 2), bbox_b[:, :2].unsqueeze(0).expand(n, m, 2))
    br = torch.min(bbox_a[:, 2:].unsqueeze(1).expand(n, m, 2), bbox_b[:, 2:].unsqueeze(0).expand(n, m, 2))

    area_i = torch.prod(br - tl + offset, dim=2) * torch.prod((tl < br).float(), dim=2)
    area_a = torch.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, dim=1)
    area_b = torch.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, dim=1)
    return area_i / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - area_i)


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3, 4]]).float()
    b = torch.tensor([[1.1, 2.2, 3.3, 4.5], [2.4, 3.6, 4.1, 5.5]])
    print(bbox_iou(a, b))


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or torch.tensor
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or torch.tensor
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is tensor, return is tensor correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = torch.clamp(xywh[2] - 1, 0), torch.clamp(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, torch.Tensor):
        if not xywh.numel() % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = torch.cat([xywh[:, :2], xywh[:, :2] + torch.clamp(xywh[:, 2:4] - 1, 0)], dim=1)
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or tensor
        The bbox in format (xmin, ymin, xmax, ymax).
        If tensor is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or tensor
        The converted bboxes in format (x, y, w, h).
        If input is tensor, return is tensor correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return x1, y1, w, h
    elif isinstance(xyxy, torch.Tensor):
        if not xyxy.numel() % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return torch.cat([xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1], dim=1)
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or tensor
        The bbox in format (xmin, ymin, xmax, ymax).
        If tensor is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return x1, y1, x2, y2
    elif isinstance(xyxy, torch.Tensor):
        if not xyxy.numel() % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = torch.clamp(xyxy[:, 0], 0, width - 1)
        y1 = torch.clamp(xyxy[:, 1], 0, height - 1)
        x2 = torch.clamp(xyxy[:, 2], 0, width - 1)
        y2 = torch.clamp(xyxy[:, 3], 0, height - 1)
        return torch.cat([x1, y1, x2, y2], dim=1)
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
