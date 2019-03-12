"""Transforms for YOLO series."""
from __future__ import absolute_import

import torch
import numpy as np
from PIL import Image

import torchvision.transforms.functional as vf
import data.transforms.utils.image_pil as timage
import data.transforms.utils.bbox as tbbox

type_map = {torch.float32: np.float32, torch.float64: np.float64}


def transform_test(imgs, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : PIL.Image or iterable of PIL.Image
        Image(s) to be transformed.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (Tensor, numpy.array) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, Image.Image):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, Image.Image), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size, mult_base=stride)
        orig_img = np.array(img).astype('uint8')
        img = vf.to_tensor(img)
        img = vf.normalize(img, mean=mean, std=std)
        tensors.append(img.unsqueeze(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
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
    imgs = [Image.open(f).convert('RGB') for f in filenames]
    return transform_test(imgs, short, max_size, stride, mean, std)


class YOLO3DefaultValTransform(object):
    """Default YOLO validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        w, h = src.size
        img = timage.imresize(src, self._width, self._height, interp=Image.BILINEAR)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = vf.to_tensor(img)
        img = vf.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(type_map[img.dtype])
