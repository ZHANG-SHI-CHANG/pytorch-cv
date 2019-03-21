"""Pyramid Scene Parsing Network"""
import os
import torch
import torch.nn.functional as F

from model.models_zoo.seg.segbase import SegBaseModel
from model.module.basic_seg import _PSPHead, _FCNHead

__all__ = ['PSPNet', 'get_psp',
           'get_psp_resnet101_coco',
           'get_psp_resnet101_voc',
           'get_psp_resnet50_ade',
           'get_psp_resnet101_ade',
           'get_psp_resnet101_citys']


class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017

    """

    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.head = _PSPHead(nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
        # print('self.crop_size', self.crop_size)

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


def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False, pretrained_base=True,
            root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    r"""Pyramid Scene Parsing Network
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
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data import datasets
    # infer number of classes
    model = PSPNet(datasets[dataset].NUM_CLASS, backbone=backbone,
                   pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('psp_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_psp_resnet101_coco(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    >>> model = get_psp_resnet101_coco(pretrained=True)
    >>> print(model)
    """
    return get_psp('coco', 'resnet101', **kwargs)


def get_psp_resnet101_voc(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    >>> model = get_psp_resnet101_voc(pretrained=True)
    >>> print(model)
    """
    return get_psp('pascal_voc', 'resnet101', **kwargs)


def get_psp_resnet50_ade(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet50', **kwargs)


def get_psp_resnet101_ade(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    >>> model = get_psp_resnet101_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet101', **kwargs)


def get_psp_resnet101_citys(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    >>> model = get_psp_resnet101_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('citys', 'resnet101', **kwargs)
