# import torch
# from collections import OrderedDict
#
# root = './coco/fcn/res101/fcn_resnet101_coco_tmp.pth'
# param = torch.load(root)
# param_new = OrderedDict([(k.split('.', 1)[-1], v) for k, v in param.items()])
#
# torch.save(param_new, './coco/fcn/res101/fcn_resnet101_coco.pth')

# def func():
#     out = list()
#     a = 10
#     for i in range(5):
#         out.append(a)
#         a += i
#     return out
#
# print(func())
#
#
# from torch import nn
#
# a = nn.Conv2d(3, 20, 1, 1)
# a.out_channels = 10
# print(a)

import logging
from utils.logger import setup_logger
from torch import nn


class Demo(object):
    def __init__(self):
        self.a = nn.Conv2d(10, 20, 3, 1, 1)
        self.b = nn.Conv2d(20, 30, 3, 1)
        setattr(self, 'heads', ['a', 'b'])
        # self.__setattr__('heads', [self.a, self.b])


demo = Demo()
for name in demo.heads:
    print(getattr(demo, name).parameters())

# class Demo(nn.Module):
#     def __init__(self, parent=None):
#         super(Demo, self).__init__()
#         if parent is None:
#             self.conv1 = nn.Conv2d(10, 20, 3, 1, 1)
#             self.conv2 = nn.Conv2d(20, 30, 3, 1, 1)
#         else:
#             self.conv1 = parent.conv1
#             self.conv2 = parent.conv2
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x
#
#
# if __name__ == '__main__':
#     import torch
#
#     demo = Demo(parent=None)
#     demo2 = Demo(parent=demo)
#     print(demo.conv1.bias)
#     print(demo2.conv1.bias)
#     demo.conv1.bias.data = torch.randn(20, 1)
#     print(demo.conv1.bias)
#     print(demo2.conv1.bias)
