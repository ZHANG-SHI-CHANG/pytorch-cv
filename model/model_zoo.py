from .models_zoo.cifarresnet import *

__all__ = ['get_model', 'get_model_list']

_models = {
    # cifar
    'cifar_resnet20_v1': cifar_resnet20_v1,
    'cifar_resnet56_v1': cifar_resnet56_v1,
    'cifar_resnet110_v1': cifar_resnet110_v1,
    'cifar_resnet20_v2': cifar_resnet20_v2,
    'cifar_resnet56_v2': cifar_resnet56_v2,
    'cifar_resnet110_v2': cifar_resnet110_v2,
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
