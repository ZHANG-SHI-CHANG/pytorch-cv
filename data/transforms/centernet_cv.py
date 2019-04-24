import cv2
import numpy as np

import torch

import data.transforms.utils.image_cv as timage


def pre_process(image, scale, input_hw=(512, 512), pad=31, fix_res=True, mean=[0.408, 0.447, 0.47],
                std=[0.289, 0.274, 0.278], flip_test=False, down_ratio=4):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    if fix_res:
        inp_height, inp_width = input_hw
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
    else:
        inp_height = (new_height | pad) + 1
        inp_width = (new_width | pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)
    trans_input = timage.get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height),
                               flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}
    return images, meta


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = timage.get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = timage.affine_transform(coords[p, 0:2], trans)
    return target_coords


def post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def load_demo(filenames, scale=1, input_hw=(512, 512), pad=31, fix_res=True, mean=[0.408, 0.447, 0.47],
              std=[0.289, 0.274, 0.278], flip_test=False, down_ratio=4):
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [cv2.imread(f) for f in filenames]
    outputs = list()
    for img in imgs:
        img_pre, meta = pre_process(img, scale, input_hw, pad, fix_res, mean, std, flip_test, down_ratio)
        outputs.append((img, img_pre, meta))
    if len(outputs) == 1:
        return outputs[0]
    return outputs
