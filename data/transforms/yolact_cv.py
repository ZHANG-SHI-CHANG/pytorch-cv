"""Transforms for YOLACT series."""

import cv2
import torch
import numpy as np
import torch.nn.functional as F

import data.transforms.utils.image_cv as timage
import data.transforms.utils.functional_cv as vf
from model.ops import mask_crop, sanitize_coordinates


def transform_test(imgs, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, np.ndarray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.imresize(img, max_size, max_size, interp=1)
        orig_img = img.astype('uint8')
        img = vf.to_tensor(img)
        img = vf.normalize(img, mean=mean, std=std)
        tensors.append(img.unsqueeze(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, max_size=550, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in filenames]
    return transform_test(imgs, max_size, mean, std)


def post_process(dets, w, h, score_thresh=0.3, lin_comb=True, interp='bilinear',
                 activate=torch.sigmoid, crop_mask=True):
    if dets is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    if score_thresh > 0:
        keep = dets['score'] > score_thresh

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    names = ['class', 'box', 'score', 'mask']
    classes, boxes, scores, masks = [dets[name] for name in names]

    if lin_comb:
        proto = dets['proto']
        masks = torch.matmul(proto, masks.t())
        masks = activate(masks)
        if crop_mask:
            masks = mask_crop(masks, boxes)
        masks = masks.permute(2, 0, 1).contiguous()
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interp, align_corners=False).squeeze(0)
        # Binarize the masks
        masks = masks.gt(0.5).float()
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()
    return classes, scores, boxes, masks
