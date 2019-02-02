import os
import mxnet as mx
from gluoncv.model_zoo import get_model
import torch

import argparse


def gluon2torch(name, gluon_path, torch_path, change_key=False, base=False):
    name = name.lower()
    model_file = os.path.join(gluon_path, name + '.params')
    if not os.path.exists(model_file):
        get_model(name, pretrained=True, root=gluon_path)
        for file in os.listdir(gluon_path):
            if '-' in file and file.split('-')[0] == name:
                os.rename(os.path.join(gluon_path, file), os.path.join(gluon_path, name + '.params'))
    if change_key:
        key_map = get_key_map(name, base)
    gluon_model_params = mx.nd.load(model_file)
    pytorch_model_params = {}
    print('Convert Gluon Model to PyTorch Model ...')
    for key, value in gluon_model_params.items():
        tensor = torch.from_numpy(value.asnumpy())
        tensor.require_grad = True
        if change_key:
            pytorch_model_params[key_map[key]] = tensor
        else:
            if 'gamma' in key:
                key = key.replace('gamma', 'weight')
            elif 'beta' in key:
                key = key.replace('beta', 'bias')
            pytorch_model_params[key] = tensor
    if not os.path.exists(torch_path):
        os.makedirs(torch_path)
    torch.save(pytorch_model_params, os.path.join(torch_path, name + '.pth'))
    print('Finished!')


def get_key_map(name, base=False):
    from model import model_zoo
    if base:
        torch_model = model_zoo.get_model(name, pretrained=False, pretrained_base=False)
        gluon_model = get_model(name, pretrained=False, pretrained_base=False)
    else:
        torch_model = model_zoo.get_model(name, pretrained=False)
        gluon_model = get_model(name, pretrained=False)
    torch_keys = [k for k in torch_model.state_dict().keys() if not k.endswith('num_batches_tracked')]
    gluon_keys = [k[len(gluon_model.name) + 1:] for k in gluon_model.collect_params().keys()]

    assert len(torch_keys) == len(gluon_keys)
    return dict(zip(gluon_keys, torch_keys))



def get_key_map(name, base=False):
    from model import model_zoo
    if base:
        torch_model = model_zoo.get_model(name, pretrained=False, pretrained_base=False)
        gluon_model = get_model(name, pretrained=False, pretrained_base=False)
    else:
        torch_model = model_zoo.get_model(name, pretrained=False)
        gluon_model = get_model(name, pretrained=False)
    torch_keys = [k for k in torch_model.state_dict().keys() if not k.endswith('num_batches_tracked')]
    gluon_keys = [k[len(gluon_model.name) + 1:] for k in gluon_model.collect_params().keys()]

    assert len(torch_keys) == len(gluon_keys)
    return dict(zip(gluon_keys, torch_keys))

if __name__ == '__main__':
    home = os.path.expanduser('~')

    parse = argparse.ArgumentParser(description='Convert gluon model to pytorch')
    parse.add_argument('--name', type=str, default='ResNet50_v1', help='name of the model')
    parse.add_argument('--gluon-path', type=str, default=os.path.join(home, '.mxnet/models'),
                       help='path to the gluon models')
    parse.add_argument('--torch-path', type=str, default=os.path.join(home, '.torch/models'),
                       help='path to the pytorch models')
    parse.add_argument('--change_key', type=bool, default=True, help='change model keys')
    parse.add_argument('--base', type=bool, default=False, help='use pretrained_base')

    config = parse.parse_args()
    gluon2torch(config.name, config.gluon_path, config.torch_path, config.change_key, config.base)
