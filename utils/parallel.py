import torch


def tuple_map(obj):
    if isinstance(obj, torch.Tensor):
        return (obj,)
    if isinstance(obj, list) and len(obj) > 0:
        return tuple(obj)
    return obj
