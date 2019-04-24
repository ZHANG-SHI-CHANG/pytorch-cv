"""Predictor for classification/box prediction."""
from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F


class ConvPredictor(nn.Module):
    """Convolutional predictor.
    Convolutional predictor is widely used in object-detection. It can be used
    to predict classification scores (1 channel per class) or box predictor,
    which is usually 4 channels per box.
    The output is of shape (N, num_channel, H, W).

    Parameters
    ----------
    in_channel : int
        Number of input channels
    channel : int
        Number of conv channels.
    kernel : tuple of (int, int), default (3, 3)
        Conv kernel size as (H, W).
    pad : tuple of (int, int), default (1, 1)
        Conv padding size as (H, W).
    stride : tuple of (int, int), default (1, 1)
        Conv stride size as (H, W).
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.

    """

    def __init__(self, in_channel, channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation=None, use_bias=True, **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        self.activation = activation
        self.predictor = nn.Conv2d(in_channel, channel, kernel,
                                   stride=stride, padding=pad, bias=use_bias)

    def forward(self, x):
        x = self.predictor(x)
        if self.activation:
            x = F.relu(x)
        return x
