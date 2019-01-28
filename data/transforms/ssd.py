"""Transforms described in https://arxiv.org/abs/1512.02325."""
from __future__ import absolute_import

import numpy as np
from PIL import Image

import torchvision.transforms.functional as vf
from data.transforms.utils.image_pil import resize_short_within


def transform_test(imgs, short, max_size=1024, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : PIL.Image or iterable of PIL.Image
        Image(s) to be transformed.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (Tensor, numpy.array) or list of such tuple
        A (1, 3, H, W) torch.Tensor as input to network, and a numpy array as
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
        img = resize_short_within(img, short, max_size)
        orig_img = np.array(img).astype('uint8')
        img = vf.to_tensor(img)
        img = vf.normalize(img, mean=mean, std=std)
        tensors.append(img.unsqueeze(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short, max_size=1024, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or iterable of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (torch.Tensor, numpy.array) or list of such tuple
        A (1, 3, H, W) torch Tensor as input to network, and a numpy array as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [Image.open(f) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)
