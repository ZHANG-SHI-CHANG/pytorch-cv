import mxnet as mx
import torch

from model import model_zoo

from gluoncv.model_zoo import get_model

model = model_zoo.get_model('resnet50_v1', pretrained=False)
l1 = model.features.children()
l = list(l1)
for k in l:
    print(k)
    print()

# model2 = get_model('resnet50_v1', pretrained=False)
# print(model2)

# model_file = '/home/ace/.mxnet/models/ssd_300_vgg16_atrous_voc.params'
# gluon_model_params = mx.nd.load(model_file)
#
# for key, _ in gluon_model_params.items():
#     print(key)