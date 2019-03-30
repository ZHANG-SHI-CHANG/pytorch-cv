# FOR vgg-astrous

import os
import sys
import argparse
import torch
from gluoncv.model_zoo import get_model

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
from model import model_zoo


def gluon2torch(name, gluon_path, torch_path, num):
    name = name.lower()
    torch_model = model_zoo.get_model(name, pretrained=False, root=torch_path)
    gluon_model = get_model(name, pretrained=True, root=gluon_path)
    torch_keys = [k for k in torch_model.state_dict().keys() if not k.endswith('num_batches_tracked')]
    gluon_keys = gluon_model.collect_params().keys()
    assert len(torch_keys) == len(gluon_keys)

    map = dict(zip(gluon_keys, torch_keys))
    pytorch_model_params = {}
    print('Convert Gluon Model to PyTorch Model ...')
    for i, ((key, value), (key2, value2)) in enumerate(
            zip(gluon_model.collect_params().items(), torch_model.state_dict().items())):
        if i < num:
            tensor = torch.from_numpy(value.data().asnumpy())
            tensor.require_grad = True
            pytorch_model_params[map[key]] = tensor
        else:
            pytorch_model_params[map[key]] = value2

    torch.save(pytorch_model_params, os.path.join(torch_path, name + '.pth'))
    print('Finish')


if __name__ == '__main__':
    home = os.path.expanduser('~')

    parse = argparse.ArgumentParser(description='Convert gluon model to pytorch')
    parse.add_argument('--name', type=str, default='vgg16_atrous_300', help='name of the model')
    parse.add_argument('--gluon-path', type=str, default=os.path.join(home, '.mxnet/models'),
                       help='path to the gluon models')
    parse.add_argument('--torch-path', type=str, default=os.path.join(home, '.torch/models'),
                       help='path to the pytorch models')
    parse.add_argument('--num', type=int, default=32,
                       help='num of layer need to convert')

    config = parse.parse_args()
    gluon2torch(config.name, config.gluon_path, config.torch_path, 32)
