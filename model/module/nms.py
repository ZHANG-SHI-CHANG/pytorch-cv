import torch


# (x, y, w, h)--->(xmin, y_min, x_max, y_max)
def point_form(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), boxes.dim() - 1)


# (xmin, y_min, x_max, y_max)--->(x, y, w, h)
def center_form(boxes):
    return torch.cat(((boxes[..., 2:] + boxes[..., :2]) / 2, boxes[..., 2:] - boxes[..., :2]), boxes.dim() - 1)


# calculate intersection area: A--[mx4], B--[nx4] --> out--[mxn](area)
# Note: point form
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    iter = torch.clamp(max_xy - min_xy, min=0)
    return iter[:, :, 0] * iter[:, :, 1]


# calculate (A∩B)/(A∪B) --- return size [mxn]
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


# nom-maximum suppression---boxes:[nx4], scores:[n]
def nms(boxes, scores, overlap=0.5, top_k=200):
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


# def box_nms(data, overlap_thresh=0.5, topk=-1, coord_start=2,
#             score_index=1, id_index=-1):
#     assert data.shape[0] == 1
#     keep, count = nms(data[0, :, coord_start:coord_start + 4], data[0, :, score_index], overlap_thresh)
#     if topk > 0:
#         keep = keep[:topk]
#     return data[:, keep, :]

def box_nms(data, overlap_thresh=0.5, topk=-1, coord_start=2,
            score_index=1, id_index=-1):
    results = list()
    b = data.shape[0]
    for i in range(b):
        keep, count = nms(data[i, :, coord_start:coord_start + 4], data[i, :, score_index], overlap_thresh)
        if topk > 0:
            keep = keep[:topk]
        results.append(data[i:i+1, keep, :])
    return torch.cat(results, 0)


if __name__ == '__main__':
    a = torch.Tensor([[2, 3, 4, 5], [3, 4, 5, 6]]).view(1, 1, 2, 4)
    print(point_form(a))
