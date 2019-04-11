# TODO: remove nms to c++ extension
import torch
from model import _C


# nom-maximum suppression---boxes:[nx4], scores:[n]
def nms_py(boxes, scores, overlap=0.5, top_k=200):
    keep = torch.zeros(scores.size(0)).long()
    if boxes.numel() == 0:
        return keep, 0
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    count = 0
    while idx.numel() > 0:
        i = idx[0]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[1:]
        xx1 = x1.index_select(0, idx).clamp_(min=x1[i].item())
        yy1 = y1.index_select(0, idx).clamp_(min=y1[i].item())
        xx2 = x2.index_select(0, idx).clamp_(max=x2[i].item())
        yy2 = y2.index_select(0, idx).clamp_(max=y2[i].item())
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        rem_areas = area.index_select(0, idx)
        union = rem_areas - inter + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]
    return keep, count


def box_nms_py(data, iou_threshold=0.5, topk=-1, coord_start=2,
               score_index=1):
    results = list()
    b = data.shape[0]
    for i in range(b):
        keep, count = nms_py(data[i, :, coord_start:coord_start + 4], data[i, :, score_index], iou_threshold)
        if topk > 0:
            keep = keep[:topk]
        results.append(data[i:i + 1, keep, :])
    return torch.cat(results, 0)


# NOTE: this two version nms is different
def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).
    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.
    Arguments:
        boxes (Tensor[N, 4]): boxes to perform NMS on
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping
            boxes with IoU < iou_threshold
    Returns:
        keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS
    """
    return _C.nms(boxes, scores, iou_threshold)


# TODO move sort to C++
def box_nms(data, iou_threshold=0.5, topk=-1, coord_start=2,
            score_index=1, score_threshold=0.01, sort=False):
    data = data[data[:, score_index] > score_threshold, :]
    if sort:
        idx = torch.argsort(data[:, score_index], descending=True)
        data = data[idx, :]
    keep = nms(data[:, coord_start:coord_start + 4], data[:, score_index], iou_threshold)
    if topk > 0:
        keep = keep[:topk]
    return data[keep, :]


# center to corner
def bbox_center_to_corner(bbox, axis=-1, split=False):
    x, y, w, h = torch.split(bbox, 1, dim=axis)
    hw = w / 2
    hh = h / 2
    xmin = x - hw
    ymin = y - hh
    xmax = x + hw
    ymax = y + hh
    if not split:
        return torch.cat([xmin, ymin, xmax, ymax], dim=axis)
    else:
        return xmin, ymin, xmax, ymax


# corner to center
def bbox_corner_to_center(bbox, axis=-1, split=False):
    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=axis)
    width = xmax - xmin
    height = ymax - ymin
    x = xmin + width / 2
    y = ymin + height / 2
    if not split:
        return torch.cat([x, y, width, height], dim=axis)
    else:
        return x, y, width, height


# box split
def bbox_split(bbox, axis, squeeze_axis=False):
    if squeeze_axis:
        return tuple(x.squeeze_(axis) for x in torch.split(bbox, 1, dim=axis))
    else:
        return torch.split(bbox, 1, dim=axis)


# iou: nx4, mx4 ---> nxm
def bbox_iou(bbox_a, bbox_b, fmt='corner', offset=0):
    if fmt.lower() == 'center':
        bbox_a = bbox_center_to_corner(bbox_a)
        bbox_b = bbox_center_to_corner(bbox_b)
    elif fmt.lower() == 'corner':
        pass
    else:
        raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")
    n, m = bbox_a.shape[0], bbox_b.shape[0]
    tl = torch.max(bbox_a[:, :2].unsqueeze(1).expand(n, m, 2), bbox_b[:, :2].unsqueeze(0).expand(n, m, 2))
    br = torch.min(bbox_a[:, 2:].unsqueeze(1).expand(n, m, 2), bbox_b[:, 2:].unsqueeze(0).expand(n, m, 2))

    area_i = torch.prod(br - tl + offset, dim=2) * torch.prod((tl < br).float(), dim=2)
    area_a = torch.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, dim=1)
    area_b = torch.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, dim=1)
    return area_i / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - area_i)


# batch iou: bxnx4, bxmx4 ---> bxnxm
def bbox_iou_batch(bbox_a, bbox_b, fmt='corner', offset=0, eps=1e-15):
    if fmt.lower() == 'center':
        al, at, ar, ab = bbox_center_to_corner(bbox_a, split=True)
        bl, bt, br, bb = bbox_center_to_corner(bbox_b, split=True)
    elif fmt.lower() == 'corner':
        al, at, ar, ab = bbox_split(bbox_a, axis=-1, squeeze_axis=True)
        bl, bt, br, bb = bbox_split(bbox_b, axis=-1, squeeze_axis=True)
    else:
        raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))
    b, n, m = bbox_b.shape[0], bbox_a.shape[1], bbox_b.shape[1]
    # (B, N, M)
    left = torch.max(al.unsqueeze(2).expand(b, n, m), bl.unsqueeze(1).expand(b, n, m))
    right = torch.min(ar.unsqueeze(2).expand(b, n, m), br.unsqueeze(1).expand(b, n, m))
    top = torch.max(at.unsqueeze(2).expand(b, n, m), bt.unsqueeze(1).expand(b, n, m))
    bot = torch.min(ab.unsqueeze(2).expand(b, n, m), bb.unsqueeze(1).expand(b, n, m))

    # clip with (0, float16.max)
    iw = torch.clamp(right - left + offset, min=0, max=6.55040e+04)
    ih = torch.clamp(bot - top + offset, min=0, max=6.55040e+04)
    i = iw * ih

    # areas
    area_a = ((ar - al + offset) * (ab - at + offset)).unsqueeze(-1)
    area_b = ((br - bl + offset) * (bb - bt + offset)).unsqueeze(-2)
    union = area_a + area_b - i

    return i / (union + eps)
