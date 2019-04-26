"""Fully Convolutional Network with Stride of 8"""
from __future__ import division

import os
import torch
import torch.nn.functional as F

from model.module.basic_seg import _FCNHead
from model.models_zoo.seg.segbase import SegBaseModel

__all__ = ['FCN', 'get_fcn',
           'get_fcn_resnet50_voc',
           'get_fcn_resnet101_voc',
           'get_fcn_resnet101_coco',
           'get_fcn_resnet101_ade',
           'get_fcn_resnet50_ade',
           'get_fcn_resnet101_citys']


class FCN(SegBaseModel):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    pretrained_base : bool or str
        Refers to if the FCN backbone or the encoder is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """

    # pylint: disable=arguments-differ
    def __init__(self, nclass, backbone='resnet50', aux=True, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(FCN, self).__init__(nclass, aux, backbone, base_size=base_size, crop_size=crop_size,
                                  dilated=dilated, jpu=jpu, pretrained_base=pretrained_base, **kwargs)
        self.head = _FCNHead(2048, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
        self.__setattr__('others', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, self._up_kwargs, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, self._up_kwargs, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root=os.path.expanduser('~/.torch/models'), pretrained_base=True, **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.

    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_paper': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys'
    }
    from data import datasets
    # infer number of classes
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base,
                **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file(
            'fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


def get_fcn_resnet50_voc(**kwargs):
    r"""FCN model with base network ResNet-101 pre-trained on Pascal VOC dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet50_voc(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pascal_voc', 'resnet50', **kwargs)


def get_fcn_resnet101_voc(**kwargs):
    r"""FCN model with base network ResNet-101 pre-trained on Pascal VOC dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet101_voc(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pascal_voc', 'resnet101', **kwargs)


def get_fcn_resnet101_coco(**kwargs):
    r"""FCN model with base network ResNet-101 pre-trained on Pascal VOC dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet101_coco(pretrained=True)
    >>> print(model)
    """
    return get_fcn('coco', 'resnet101', **kwargs)


def get_fcn_resnet101_ade(**kwargs):
    r"""FCN model with base network ResNet-50 pre-trained on ADE20K dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn('ade20k', 'resnet101', **kwargs)


def get_fcn_resnet50_ade(**kwargs):
    r"""FCN model with base network ResNet-50 pre-trained on ADE20K dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn('ade20k', 'resnet50', **kwargs)


def get_fcn_resnet101_citys(**kwargs):
    r"""FCN model with base network ResNet-50 pre-trained on ADE20K dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn('citys', 'resnet101', **kwargs)