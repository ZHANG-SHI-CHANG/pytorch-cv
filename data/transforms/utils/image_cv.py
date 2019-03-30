"""Extended image transformations to `torchvision`."""
from __future__ import division

import random
import numpy as np
import cv2
import data.transforms.utils.functional_cv as vf
from data.transforms.utils.functional_cv import get_interp_method


def imresize(src, w, h, interp=1):
    """Resize image."""
    oh, ow, _ = src.shape
    return vf.resize(src, (w, h), interpolation=get_interp_method(interp, (oh, ow, h, w)))


def resize_short_within(img, short, max_size, mult_base=1, interp=2):
    """Resizes shorter edge to size but make sure it's capped at maximum size."""
    h, w, _ = img.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return vf.resize(img, (new_h, new_w), interp, get_interp_method(interp, (h, w, new_h, new_w)))


numeric_types = (float, int, np.generic)


def random_expand(src, max_ratio=4, fill=0, keep_ratio=True):
    """Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with HWC format.
    max_ratio : int or float
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill : int or float or array-like
        The value(s) for padded borders. If `fill` is numerical type, RGB channels
        will be padded with single value. Otherwise `fill` must have same length
        as image channels, which resulted in padding with per-channel values.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (offset_x, offset_y, new_width, new_height)

    """
    if max_ratio <= 1:
        return src, (0, 0, src.shape[1], src.shape[0])

    h, w, c = src.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    # make canvas
    if isinstance(fill, numeric_types):
        dst = np.full(shape=(oh, ow, c), val=fill, dtype=src.dtype)
    else:
        fill = np.array(fill, dtype=src.dtype)
        if not c == fill.size:
            raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
        dst = np.tile(fill.reshape((1, c)), reps=(oh * ow, 1)).reshape((oh, ow, c))

    dst[off_y:off_y + h, off_x:off_x + w, :] = src
    return dst, (off_x, off_y, ow, oh)


def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size.

    Parameters
    ----------
    src : NDArray
        Input image
    x0 : int
        Left boundary of the cropping area
    y0 : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : tuple of (w, h)
        Optional, resize to new size after cropping
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    """
    out = src[y0:y0 + h, x0:x0 + w, :]
    if size is not None and (w, h) != size:
        sizes = (h, w, size[1], size[0])
        out = cv2.resize(out, *size, interpolation=get_interp_method(interp, sizes))
    return out


def random_flip(src, px=0, py=0, copy=False):
    """Randomly flip image along horizontal and vertical with probabilities.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image with HWC format.
    px : float
        Horizontal flip probability [0, 1].
    py : float
        Vertical flip probability [0, 1].
    copy : bool
        If `True`, return a copy of input

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (flip_x, flip_y), records of whether flips are applied.

    """
    flip_y = np.random.choice([False, True], p=[1 - py, py])
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    if flip_y:
        src = np.flip(src, axis=0)
    if flip_x:
        src = np.flip(src, axis=1)
    if copy:
        src = src.copy()
    return src, (flip_x, flip_y)
