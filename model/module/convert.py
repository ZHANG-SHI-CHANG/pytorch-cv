from torch import nn


def convert_norm_layer(module, norm_layer, norm_kwargs):
    module_output = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module_output = norm_layer(module.num_features, **norm_kwargs)
    for name, child in module.named_children():
        module_output.add_module(name, convert_norm_layer(child, norm_layer, norm_kwargs))
    del module
    return module_output
