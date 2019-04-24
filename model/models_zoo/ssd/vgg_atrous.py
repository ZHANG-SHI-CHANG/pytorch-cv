"""VGG atrous network for object detection."""
# TODO: change to SyncBatch
from __future__ import division

import os
import torch
from torch import nn
import torch.nn.functional as F

from model.module.basic import _init_scale
from utils.init import xavier_uniform_init, mxnet_xavier_normal_init

__all__ = ['VGGAtrousExtractor', 'get_vgg_atrous_extractor', 'vgg16_atrous_300',
           'vgg16_atrous_512']


class L2Norm(nn.Module):
    """Normalize layer described in https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    n_channels : int
        Number of channels of input.
    initial : float
        Initial value for the rescaling factor.
    eps : float
        Small value to avoid division by zero.
    """

    def __init__(self, n_channels, initial=20, eps=1e-5):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.eps = eps
        scale = nn.Parameter(torch.Tensor(1, self.n_channels, 1, 1))
        nn.init.constant_(scale, initial)
        self.register_parameter('normalize_scale', scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.normalize_scale * x
        return out


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class VGGAtrousBase(nn.Module):
    """VGG Atrous multi layer base network. You must inherit from it to define
    how the features are computed.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    batch_norm : bool, default is False
        If `True`, will use BatchNorm layers.
    """

    def __init__(self, layers, filters, batch_norm=False, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, **kwargs):
        super(VGGAtrousBase, self).__init__(**kwargs)
        assert len(layers) == len(filters) - 1

        # we use pre-trained weights from caffe, initial scale must change
        self.register_parameter('init_scale', _init_scale([0.229, 0.224, 0.225]))

        self.stages = list()
        for i in range(len(layers)):
            in_channels, channels = filters[i], filters[i + 1]
            stage = list()
            for _ in range(layers[i]):
                stage.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
                if batch_norm:
                    stage.append(norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)))
                stage.append(nn.ReLU(inplace=True))
                in_channels = channels
            self.stages.append(nn.Sequential(*stage))

        # use dilated convolution instead of dense layers
        stage = list()
        stage.append(nn.Conv2d(filters[-1], 1024, kernel_size=3, padding=6, dilation=6))
        if batch_norm:
            stage.append(norm_layer(1024, **({} if norm_kwargs is None else norm_kwargs)))
        stage.append(nn.ReLU(inplace=True))
        stage.append(nn.Conv2d(1024, 1024, kernel_size=1))
        if batch_norm:
            stage.append(norm_layer(1024, **({} if norm_kwargs is None else norm_kwargs)))
        stage.append(nn.ReLU(inplace=True))
        self.stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*self.stages)

        # normalize layer for 4-th stage
        self.norm4 = L2Norm(filters[-2], 20)

    def forward(self, x):
        raise NotImplementedError


class VGGAtrousExtractor(VGGAtrousBase):
    """VGG Atrous multi layer feature extractor which produces multiple output
    feature maps.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    extras : list of list
        Extra layers configurations.
    batch_norm : bool
        If `True`, will use BatchNorm layers.

    """

    def __init__(self, layers, filters, extras, channel=[512, 1024], batch_norm=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(VGGAtrousExtractor, self).__init__(layers, filters, batch_norm, norm_layer, norm_kwargs, **kwargs)
        self.extras = list()
        self.channel = channel
        for i, config in enumerate(extras):
            extra = list()
            for f_in, f, k, s, p in config:
                extra.append(nn.Conv2d(f_in, f, k, s, p))
                if batch_norm:
                    extra.append(norm_layer(f, **({} if norm_kwargs is None else norm_kwargs)))
                extra.append(nn.ReLU(inplace=True))
            self.channel.append(f)
            self.extras.append(nn.Sequential(*extra))
        self.extras = nn.Sequential(*self.extras)
        self._weight_init()

    def forward(self, x):
        x = x * self.init_scale
        assert len(self.stages) == 6
        outputs = list()
        for stage in self.stages[:3]:
            x = stage(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = self.stages[3](x)
        norm = self.norm4(x)
        outputs.append(norm)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = self.stages[4](x)
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=(1, 1), ceil_mode=True)
        x = self.stages[5](x)
        outputs.append(x)
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        return outputs

    def _weight_init(self):
        # self.stages.apply(xavier_uniform_init)
        # self.extras.apply(xavier_uniform_init)
        self.stages.apply(mxnet_xavier_normal_init)
        self.extras.apply(mxnet_xavier_normal_init)


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
vgg_spec = {
    11: ([1, 1, 2, 2, 2], [3, 64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [3, 64, 128, 256, 512, 512]),
    16: ([2, 2, 3, 3, 3], [3, 64, 128, 256, 512, 512]),
    19: ([2, 2, 4, 4, 4], [3, 64, 128, 256, 512, 512])
}

extra_spec = {
    300: [((1024, 256, 1, 1, 0), (256, 512, 3, 2, 1)),
          ((512, 128, 1, 1, 0), (128, 256, 3, 2, 1)),
          ((256, 128, 1, 1, 0), (128, 256, 3, 1, 0)),
          ((256, 128, 1, 1, 0), (128, 256, 3, 1, 0))],

    512: [((1024, 256, 1, 1, 0), (256, 512, 3, 2, 1)),
          ((512, 128, 1, 1, 0), (128, 256, 3, 2, 1)),
          ((256, 128, 1, 1, 0), (128, 256, 3, 2, 1)),
          ((256, 128, 1, 1, 0), (128, 256, 3, 2, 1)),
          ((256, 128, 1, 1, 0), (128, 256, 4, 1, 1))],
}


def get_vgg_atrous_extractor(num_layers, im_size, pretrained=False,
                             root=os.path.join(os.path.expanduser('~'), '.torch', 'models'), **kwargs):
    """Get VGG atrous feature extractor networks.

    Parameters
    ----------
    num_layers : int
        VGG types, can be 11,13,16,19.
    im_size : int
        VGG detection input size, can be 300, 512.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str
        Model weights storing path.

    Returns
    -------
    nn.Module
        The returned network.

    """
    layers, filters = vgg_spec[num_layers]
    extras = extra_spec[im_size]
    net = VGGAtrousExtractor(layers, filters, extras, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.load_state_dict(torch.load(get_model_file('vgg%d_atrous%s_%d' % (num_layers, batch_norm_suffix, im_size),
                                                      root=root)))
    return net


def vgg16_atrous_300(**kwargs):
    """Get VGG atrous 16 layer 300 in_size feature extractor networks."""
    return get_vgg_atrous_extractor(16, 300, **kwargs)


def vgg16_atrous_512(**kwargs):
    """Get VGG atrous 16 layer 512 in_size feature extractor networks."""
    return get_vgg_atrous_extractor(16, 512, **kwargs)


if __name__ == '__main__':
    net = vgg16_atrous_300(pretrained=True)
    import numpy as np
    np.random.seed(10)

    a = np.random.randn(1, 3, 300, 300).astype(np.float32)
    a = torch.from_numpy(a)

    with torch.no_grad():
        out = net(a)

    print(out[2], out[2].shape)

    # a = torch.randn(1, 3, 300, 300)
    # print(len(net(a)))
