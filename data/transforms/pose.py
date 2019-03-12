import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as F


def upscale_bbox_fn(bbox, img, scale=1.25):
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def crop_resize_normalize(img, bbox_list, output_size):
    output_list = []
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for bbox in bbox_list:
        x0 = max(int(bbox[0]), 0)
        y0 = max(int(bbox[1]), 0)
        x1 = min(int(bbox[2]), int(img.shape[1]))
        y1 = min(int(bbox[3]), int(img.shape[0]))
        w = x1 - x0
        h = y1 - y0
        res_img = F.resized_crop(F.to_pil_image(img), y0, x0, h, w, (output_size[0], output_size[1]))
        res_img = transform_test(res_img)
        output_list.append(res_img)
    output_array = torch.stack(output_list, 0)
    return output_array


def detector_to_simple_pose(img, class_IDs, scores, bounding_boxs,
                            output_shape=(256, 192), scale=1.25, device=torch.device('cpu')):
    L = class_IDs.shape[1]
    thr = 0.5
    upscale_bbox = []
    for i in range(L):
        if class_IDs[0][i].item() != 0:
            continue
        if scores[0][i].item() < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.cpu().numpy().tolist(), img, scale=scale))
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape)
        pose_input = pose_input.to(device)
    else:
        pose_input = None
    return pose_input, upscale_bbox


def get_max_pred(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals, _ = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2).float()

    preds *= pred_mask
    return preds, maxvals


def heatmap_to_coord(heatmaps, bbox_list):
    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]
    coords, maxvals = get_max_pred(heatmaps)
    preds = torch.zeros_like(coords)

    for i, bbox in enumerate(bbox_list):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        w = (x1 - x0) / 2
        h = (y1 - y0) / 2
        center = np.array([x0 + w, y0 + h])
        scale = np.array([w, h])

        w_ratio = coords[i][:, 0] / heatmap_width
        h_ratio = coords[i][:, 1] / heatmap_height
        preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
        preds[i][:, 1] = scale[1] * 2 * h_ratio + center[1] - scale[1]
    return preds, maxvals
