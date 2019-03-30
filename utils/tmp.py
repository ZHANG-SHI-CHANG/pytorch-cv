import torch
from collections import OrderedDict

root = './coco/fcn/res101/fcn_resnet101_coco_tmp.pth'
param = torch.load(root)
param_new = OrderedDict([(k.split('.', 1)[-1], v) for k, v in param.items()])

torch.save(param_new, './coco/fcn/res101/fcn_resnet101_coco.pth')
