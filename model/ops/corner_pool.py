from torch.autograd import Function
from torch.autograd.function import once_differentiable

from model import _C


class _TopPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = _C.top_pool_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.top_pool_backward(input, grad_output)[0]
        return output


class _BottomPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = _C.bottom_pool_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.bottom_pool_backward(input, grad_output)[0]
        return output


class _LeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = _C.left_pool_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.left_pool_backward(input, grad_output)[0]
        return output


class _RightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = _C.right_pool_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.right_pool_backward(input, grad_output)[0]
        return output


def top_pool(x):
    return _TopPoolFunction.apply(x)


def bottom_pool(x):
    return _BottomPoolFunction.apply(x)


def left_pool(x):
    return _LeftPoolFunction.apply(x)


def right_pool(x):
    return _RightPoolFunction.apply(x)


if __name__ == '__main__':
    import torch
    a = torch.randn(1, 1, 30, 40).cuda()
    out = top_pool(a)
    print(out)