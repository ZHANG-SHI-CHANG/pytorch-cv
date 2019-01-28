import mxnet as mx

model_file = '/home/ace/.mxnet/models/ssd_300_vgg16_atrous_voc.params'
gluon_model_params = mx.nd.load(model_file)

for key, _ in gluon_model_params.items():
    print(key)