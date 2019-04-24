import math
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out


def mxnet_xavier_(tensor, rnd_type='uniform', mode='avg', magnitude=3):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "avg":
        factor = (fan_in + fan_out) / 2.0
    elif mode == "in":
        factor = fan_in
    elif mode == "out":
        factor = fan_out
    else:
        raise ValueError("Incorrect factor type")
    scale = math.sqrt(magnitude / factor)
    with torch.no_grad():
        if rnd_type == 'uniform':
            return tensor.uniform_(-scale, scale)
        elif rnd_type == 'normal':
            return tensor.normal_(0, scale)
        else:
            raise ValueError("Unknown random type")
