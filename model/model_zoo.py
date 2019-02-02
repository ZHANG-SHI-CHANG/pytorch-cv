from .models_zoo.cifarresnet import *
from .models_zoo.cifarwideresnet import *
from .models_zoo.cifarresnext import *
from .models_zoo.vgg import *
from .models_zoo.resnet import *
from .models_zoo.ssd.ssd import *

__all__ = ['get_model', 'get_model_list']

_models = {
    # cifar
    'cifar_resnet20_v1': cifar_resnet20_v1,
    'cifar_resnet56_v1': cifar_resnet56_v1,
    'cifar_resnet110_v1': cifar_resnet110_v1,
    'cifar_resnet20_v2': cifar_resnet20_v2,
    'cifar_resnet56_v2': cifar_resnet56_v2,
    'cifar_resnet110_v2': cifar_resnet110_v2,
    'cifar_wideresnet16_10': cifar_wideresnet16_10,
    'cifar_wideresnet28_10': cifar_wideresnet28_10,
    'cifar_wideresnet40_8': cifar_wideresnet40_8,
    'cifar_resnext29_32x4d': cifar_resnext29_32x4d,
    'cifar_resnext29_16x64d': cifar_resnext29_16x64d,
    # imagenet - resnet
    'resnet18_v1': resnet18_v1,
    'resnet34_v1': resnet34_v1,
    'resnet50_v1': resnet50_v1,
    'resnet101_v1': resnet101_v1,
    'resnet152_v1': resnet152_v1,
    'resnet18_v2': resnet18_v2,
    'resnet34_v2': resnet34_v2,
    'resnet50_v2': resnet50_v2,
    'resnet101_v2': resnet101_v2,
    'resnet152_v2': resnet152_v2,
    # imagenet - vgg
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    # voc + ssd
    'ssd_300_vgg16_atrous_voc': ssd_300_vgg16_atrous_voc,
    'ssd_512_vgg16_atrous_voc': ssd_512_vgg16_atrous_voc,
    'ssd_512_resnet50_v1_voc': ssd_512_resnet50_v1_voc,

}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % name
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
