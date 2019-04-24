"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import random
from .image import plot_image

default_color = {0: (0.466, 0.674, 0.188),
                 1: (0.85, 0.325, 0.098),
                 2: (0.929, 0.694, 0.125),
                 3: (0.494, 0.184, 0.556),
                 4: (0.3, 0.4, 0.4),
                 5: (0.301, 0.745, 0.933),
                 6: (0.635, 0.078, 0.184),
                 7: (0.3, 0.3, 0.3),
                 8: (0.6, 0.6, 0.6),
                 9: (1.0, 0.0, 0.0),
                 10: (1.0, 0.5, 0.0),
                 11: (0.749, 0.749, 0.0),
                 12: (0.0, 1.0, 0.0),
                 13: (0.0, 0.0, 1.0),
                 14: (0.667, 0.0, 1.0),
                 15: (0.333, 0.333, 0.0),
                 16: (0.333, 0.667, 0.0),
                 17: (0.333, 1.0, 0.0),
                 18: (0.667, 0.333, 0.0),
                 19: (0.667, 0.667, 0.0),
                 20: (0.667, 1.0, 0.0),
                 21: (1.0, 0.333, 0.0),
                 22: (1.0, 0.667, 0.0),
                 23: (1.0, 1.0, 0.0),
                 24: (0.0, 0.333, 0.5),
                 25: (0.0, 0.667, 0.5),
                 26: (0.0, 1.0, 0.5),
                 27: (0.333, 0.0, 0.5),
                 28: (0.333, 0.333, 0.5),
                 29: (0.333, 0.667, 0.5),
                 30: (0.333, 1.0, 0.5),
                 31: (0.667, 0.0, 0.5),
                 32: (0.667, 0.333, 0.5),
                 33: (0.667, 0.667, 0.5),
                 34: (0.667, 1.0, 0.5),
                 35: (1.0, 0.0, 0.5),
                 36: (1.0, 0.333, 0.5),
                 37: (1.0, 0.667, 0.5),
                 38: (1.0, 1.0, 0.5),
                 39: (0.0, 0.333, 1.0),
                 40: (0.0, 0.667, 1.0),
                 41: (0.0, 1.0, 1.0),
                 42: (0.333, 0.0, 1.0),
                 43: (0.333, 0.333, 1.0),
                 44: (0.333, 0.667, 1.0),
                 45: (0.333, 1.0, 1.0),
                 46: (0.667, 0.0, 1.0),
                 47: (0.667, 0.333, 1.0),
                 48: (0.667, 0.667, 1.0),
                 49: (0.667, 1.0, 1.0),
                 50: (1.0, 0.0, 1.0),
                 51: (1.0, 0.333, 1.0),
                 52: (1.0, 0.667, 1.0),
                 53: (0.167, 0.0, 0.0),
                 54: (0.333, 0.0, 0.0),
                 55: (0.5, 0.0, 0.0),
                 56: (0.667, 0.0, 0.0),
                 57: (0.833, 0.0, 0.0),
                 58: (1.0, 0.0, 0.0),
                 59: (0.0, 0.167, 0.0),
                 60: (0.0, 0.333, 0.0),
                 61: (0.0, 0.5, 0.0),
                 62: (0.0, 0.667, 0.0),
                 63: (0.0, 0.833, 0.0),
                 64: (0.0, 1.0, 0.0),
                 65: (0.0, 0.0, 0.167),
                 66: (0.0, 0.0, 0.333),
                 67: (0.0, 0.0, 0.5),
                 68: (0.0, 0.0, 0.667),
                 69: (0.0, 0.0, 0.833),
                 70: (0.0, 0.0, 1.0),
                 71: (0.0, 0.0, 0.0),
                 72: (0.143, 0.143, 0.143),
                 73: (0.286, 0.286, 0.286),
                 74: (0.429, 0.429, 0.429),
                 75: (0.571, 0.571, 0.571),
                 76: (0.714, 0.714, 0.714),
                 77: (0.857, 0.857, 0.857),
                 78: (0.0, 0.447, 0.741),
                 79: (0.5, 0.5, 0.0), }


# TODO: change colors to make more visible
def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=default_color, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.array or torch.Tensor
        Image with shape `H, W, 3`.
    bboxes : numpy.array
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.array, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    # if isinstance(bboxes, mx.nd.NDArray):
    #     bboxes = bboxes.asnumpy()
    # if isinstance(labels, mx.nd.NDArray):
    #     labels = labels.asnumpy()
    # if isinstance(scores, mx.nd.NDArray):
    #     scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=2.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=11, color='white')
    return ax
