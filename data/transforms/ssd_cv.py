"""Transforms described in https://arxiv.org/abs/1512.02325."""
from __future__ import absolute_import

import torch
import numpy as np
import cv2

import data.transforms.utils.functional_cv as vf
import data.transforms.utils.image_cv as timage
import data.transforms.utils.bbox as tbbox
import data.transforms.experimental.image as eximage
import data.transforms.experimental.bbox as exbbox

type_map = {torch.float32: np.float32, torch.float64: np.float64}


def transform_test(imgs, short, max_size=1024, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 Image or iterable of Image.

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
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, np.ndarray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.astype('uint8')
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
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)


class SSDDefaultValTransform(object):
    """Default SSD validation transform.

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
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = vf.to_tensor(img)
        img = vf.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(type_map[img.dtype])


class SSDDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : torch.Tensor, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """

    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        from model.models_zoo.ssd.target import SSDTargetGenerator
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = eximage.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = exbbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = timage.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        # TODO: check it
        # img = vf.to_tensor(img / 255.)
        img = vf.to_tensor(img.astype(np.uint8))
        img = vf.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = torch.from_numpy(bbox[:, :4])
        gt_ids = torch.from_numpy(bbox[:, 4])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0]

# from data.transforms.utils.augment_cv import TrainAugmentation
#
#
# class SSDDefaultTrainTransform(object):
#     """Default SSD training transform which includes tons of image augmentations.
#
#     Parameters
#     ----------
#     width : int
#         Image width.
#     height : int
#         Image height.
#     anchors : torch.Tensor, optional
#         Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
#         Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
#         ``N`` is the number of anchors for each image.
#
#         .. hint::
#
#             If anchors is ``None``, the transformation will not generate training targets.
#             Otherwise it will generate training targets to accelerate the training phase
#             since we push some workload to CPU workers instead of GPUs.
#
#     mean : array-like of size 3
#         Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
#     std : array-like of size 3
#         Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
#     iou_thresh : float
#         IOU overlap threshold for maximum matching, default is 0.5.
#     box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
#         Std value to be divided from encoded values.
#
#     """
#
#     def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
#                  std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
#                  **kwargs):
#         self._width = width
#         self._height = height
#         self._anchors = anchors
#         if anchors is None:
#             return
#
#         self.augment = TrainAugmentation(width, mean, std)
#         # since we do not have predictions yet, so we ignore sampling here
#         from model.models_zoo.ssd.target import SSDTargetGenerator
#         self._target_generator = SSDTargetGenerator(
#             iou_thresh=iou_thresh, stds=box_norm, **kwargs)
#
#     def __call__(self, src, target):
#         """Apply transform to training image/label."""
#         boxes, labels = target[:, :4], target[:, 4]
#         src, boxes, labels = self.augment(src, boxes, labels)
#
#         if self._anchors is None:
#             return src, boxes.astype(src.dtype)
#
#         # generate training target so cpu workers can help reduce the workload on gpu
#         boxes = torch.from_numpy(boxes)
#         labels = torch.from_numpy(labels)
#         cls_targets, box_targets, _ = self._target_generator(
#             self._anchors, boxes, labels)
#         return src, cls_targets[0], box_targets[0]
