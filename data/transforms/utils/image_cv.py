"""Extended image transformations to `torchvision`."""
from __future__ import division

import numpy as np
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
