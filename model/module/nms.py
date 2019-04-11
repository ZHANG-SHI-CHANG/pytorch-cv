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





