import torch


def tuple_map(obj):
    if isinstance(obj, torch.Tensor):
        return (obj,)
    if isinstance(obj, list) and len(obj) > 0:
        return tuple(obj)
    return obj


# def split_load_kwargs(inputs, kwargs, ctx_list, batch_axis=0):
#     r"""Split with support for kwargs dictionary"""
#
#     def split_map(obj):
#         if isinstance(obj, NDArray):
#             return split_and_load(obj, ctx_list, batch_axis, even_split=False)
#         if isinstance(obj, tuple) and len(obj) > 0:
#             return list(zip(*map(split_map, obj)))
#         if isinstance(obj, list) and len(obj) > 0:
#             return list(map(list, zip(*map(split_map, obj))))
#         if isinstance(obj, dict) and len(obj) > 0:
#             return list(map(type(obj), zip(*map(split_map, obj.items()))))
#         return [obj for _ in ctx_list]
#
#     inputs = split_map(inputs) if inputs else []
#     kwargs = split_map(kwargs) if kwargs else []
#     if len(inputs) < len(kwargs):
#         inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
#     elif len(kwargs) < len(inputs):
#         kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
#     inputs = tuple(inputs)
#     kwargs = tuple(kwargs)
#     return inputs, kwargs

#
# class DataParallelModel(object):
#     """Data parallelism
#
#     Hide the difference of single/multiple GPUs to the user.
#     Inputs and outputs are both list of NDArrays in different contexts.
#     In the forward pass, the module is replicated on each device,
#     and each replica handles a portion of the input. During the backwards
#     pass, gradients from each replica are summed into the original module.
#
#     Parameters
#     ----------
#     module : object
#         Network to be parallelized.
#     ctx_list : list
#         A list of contexts
#     sync : bool
#         enable synchronization (default: False).
#
#
#     Inputs:
#         - **inputs**: list of input (NDArrays)
#
#     Outputs:
#         - **outputs**: list of output (NDArrays)
#
#     Example::
#         >>> ctx = [mx.gpu(0), mx.gpu(1)]
#         >>> net = DataParallelModel(model, ctx_list=ctx)
#         >>> y = net(x)
#     """
#
#     def __init__(self, module, device=torch.device('cpu')):
#         self.module = module.to(device)
#
#     def __call__(self, *inputs, **kwargs):
#         if not self.ctx_list:
#             return self.module(*inputs, **kwargs)
#         inputs, kwargs = split_load_kwargs(inputs, kwargs, self.ctx_list)
#         assert (len(inputs) == len(self.ctx_list))
#         if len(self.ctx_list) == 1:
#             return tuple([tuple_map(self.module(*inputs[0], **kwargs[0]))])
#         return parallel_apply(self.module, inputs, kwargs, self.sync)
#
#     def __repr__(self):
#         return 'DataParallel:\n module = {' + self.module.__repr__() + '}'
