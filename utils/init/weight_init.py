from torch import nn
from utils.init import mxnet_init


def xavier_uniform_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def mxnet_xavier_uniform_init(m, mode='avg', magnitude=2):
    if isinstance(m, nn.Conv2d):
        mxnet_init.mxnet_xavier_(m.weight, rnd_type='uniform', mode=mode, magnitude=magnitude)
        nn.init.zeros_(m.bias)


def mxnet_xavier_normal_init(m, mode='out', magnitude=2):
    if isinstance(m, nn.Conv2d):
        mxnet_init.mxnet_xavier_(m.weight, rnd_type='normal', mode=mode, magnitude=magnitude)
        nn.init.zeros_(m.bias)
