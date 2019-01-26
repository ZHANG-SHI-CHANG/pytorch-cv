import os
import mxnet as mx
from gluoncv.model_zoo import get_model
import torch

import argparse


def gluon2torch(name, gluon_path, torch_path):
    model_file = os.path.join(gluon_path, name + '.params')
    if not os.path.exists(model_file):
        get_model(name, pretrained=True, root=gluon_path)
        for file in gluon_path:
            if '-' in file and file.split('-')[0] == name:
                os.rename(os.path.join(gluon_path, file), os.path.join(gluon_path, name + '.params'))
    gluon_model_params = mx.nd.load(model_file)
    pytorch_model_params = {}
    print('Convert Gluon Model to PyTorch Model ...')
    for key, value in gluon_model_params.items():
        if 'gamma' in key:
            key = key.replace('gamma', 'weight')
        elif 'beta' in key:
            key = key.replace('beta', 'bias')

        tensor = torch.from_numpy(value.asnumpy())
        tensor.require_grad = True
        pytorch_model_params[key] = tensor

    torch.save(pytorch_model_params, os.path.join(torch_path, name + '.pth'))
    print('Finished!')


if __name__ == '__main__':
    home = os.path.expanduser('~')

    parse = argparse.ArgumentParser(description='Convert gluon model to pytorch')
    parse.add_argument('--name', type=str, default='cifar_resnet110_v1', help='name of the model')
    parse.add_argument('--gluon-path', type=str, default=os.path.join(home, '.mxnet/models'),
                       help='path to the gluon models')
    parse.add_argument('--torch-path', type=str, default=os.path.join(home, '.torch/models'),
                       help='path to the pytorch models')

    config = parse.parse_args()
    gluon2torch(config.name, config.gluon_path, config.torch_path)
