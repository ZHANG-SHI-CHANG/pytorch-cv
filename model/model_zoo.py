from .models_zoo.cifarresnet import *
from .models_zoo.cifarwideresnet import *
from .models_zoo.cifarresnext import *
from .models_zoo.vgg import *
from .models_zoo.resnet import *
from .models_zoo.resnetv1b import *
from .models_zoo.mobilenet import *
from .models_zoo.squeezenet import *
from .models_zoo.densenet import *
from .models_zoo.alexnet import *
from .models_zoo.senet import *
from .models_zoo.inception import *
from .models_zoo.yolo.darknet import *
from .models_zoo.ssd.ssd import *
from .models_zoo.yolo.yolo3 import *
from .models_zoo.seg.fcn import *
from .models_zoo.seg.pspnet import *
from .models_zoo.seg.deeplabv3 import *

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
    'resnet18_v1b': resnet18_v1b,
    'resnet34_v1b': resnet34_v1b,
    'resnet50_v1b': resnet50_v1b,
    'resnet101_v1b': resnet101_v1b,
    'resnet152_v1b': resnet152_v1b,
    'resnet50_v1c': resnet50_v1c,
    'resnet101_v1c': resnet101_v1c,
    'resnet152_v1c': resnet152_v1c,
    'resnet50_v1d': resnet50_v1d,
    'resnet101_v1d': resnet101_v1d,
    'resnet152_v1d': resnet152_v1d,
    # imagenet - mobilenet
    'mobilenet1.0': mobilenet1_0,
    'mobilenet0.75': mobilenet0_75,
    'mobilenet0.5': mobilenet0_5,
    'mobilenet0.25': mobilenet0_25,
    'mobilenetv2_1.0': mobilenet_v2_1_0,
    'mobilenetv2_0.75': mobilenet_v2_0_75,
    'mobilenetv2_0.5': mobilenet_v2_0_5,
    'mobilenetv2_0.25': mobilenet_v2_0_25,
    # imagenet - vgg
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    # imagenet - squeezenet
    'squeezenet1.0': squeezenet1_0,
    'squeezenet1.1': squeezenet1_1,
    # imagenet - densenet
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    # imagenet - others
    'alexnet': alexnet,
    'darknet53': darknet53,
    'inceptionv3': inception_v3,
    'senet_154': senet_154,
    # ssd
    'ssd_300_vgg16_atrous_voc': ssd_300_vgg16_atrous_voc,
    'ssd_512_vgg16_atrous_voc': ssd_512_vgg16_atrous_voc,
    'ssd_512_resnet50_v1_voc': ssd_512_resnet50_v1_voc,
    'ssd_512_mobilenet1.0_voc': ssd_512_mobilenet1_0_voc,
    'ssd_300_vgg16_atrous_coco': ssd_300_vgg16_atrous_coco,
    'ssd_512_vgg16_atrous_coco': ssd_512_vgg16_atrous_coco,
    'ssd_512_resnet50_v1_coco': ssd_512_resnet50_v1_coco,
    'ssd_512_mobilenet1.0_coco': ssd_512_mobilenet1_0_coco,
    # yolo3
    'yolo3_darknet53_voc': yolo3_darknet53_voc,
    'yolo3_mobilenet1.0_voc': yolo3_mobilenet1_0_voc,
    'yolo3_darknet53_coco': yolo3_darknet53_coco,
    'yolo3_mobilenet1.0_coco': yolo3_mobilenet1_0_coco,
    # fcn
    'fcn_resnet101_voc': get_fcn_resnet101_voc,
    'fcn_resnet101_coco': get_fcn_resnet101_coco,
    'fcn_resnet101_ade': get_fcn_resnet101_ade,
    'fcn_resnet50_ade': get_fcn_resnet50_ade,
    # pspnet
    'psp_resnet101_coco': get_psp_resnet101_coco,
    'psp_resnet101_voc': get_psp_resnet101_voc,
    'psp_resnet50_ade': get_psp_resnet50_ade,
    'psp_resnet101_ade': get_psp_resnet101_ade,
    'psp_resnet101_citys': get_psp_resnet101_citys,
    # deeplab
    'deeplab_resnet101_coco': get_deeplab_resnet101_coco,
    'deeplab_resnet101_voc': get_deeplab_resnet101_voc,
    'deeplab_resnet152_voc': get_deeplab_resnet152_voc,
    'deeplab_resnet50_ade': get_deeplab_resnet50_ade,
    'deeplab_resnet101_ade': get_deeplab_resnet101_ade,
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
