# import torch
# from collections import OrderedDict
#
# root = './coco/fcn/res101/fcn_resnet101_coco_tmp.pth'
# param = torch.load(root)
# param_new = OrderedDict([(k.split('.', 1)[-1], v) for k, v in param.items()])
#
# torch.save(param_new, './coco/fcn/res101/fcn_resnet101_coco.pth')

def func():
    out = list()
    a = 10
    for i in range(5):
        out.append(a)
        a += i
    return out

print(func())


from torch import nn

a = nn.Conv2d(3, 20, 1, 1)
a.out_channels = 10
print(a)