"""Extended image transformations to `torchvision`."""
from __future__ import division

from PIL import Image
import numpy as np
import torchvision.transforms.functional as vf


def imresize(src, w, h, interp=Image.BILINEAR):
    """Resize image."""
    return vf.resize(src, (h, w), interp)


def resize_short_within(img, short, max_size, mult_base=1, interp=Image.BILINEAR):
    """Resizes shorter edge to size but make sure it's capped at maximum size."""
    w, h = img.size
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return vf.resize(img, (new_h, new_w), interp)
