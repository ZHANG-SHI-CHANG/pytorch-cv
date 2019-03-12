import os
import argparse
import torch
from gluoncv.model_zoo import get_model

from model import model_zoo


def gluon2torch(name, gluon_path, torch_path, base=False, reorder=False):
    name = name.lower()
    if base:
        torch_model = model_zoo.get_model(name, pretrained=False, pretrained_base=False)
        gluon_model = get_model(name, pretrained=True, pretrained_base=False)
    else:
        torch_model = model_zoo.get_model(name, pretrained=False, root=torch_path)
        gluon_model = get_model(name, pretrained=True, root=gluon_path)
    torch_keys = [k for k in torch_model.state_dict().keys() if not k.endswith('num_batches_tracked')]
    gluon_keys = gluon_model.collect_params().keys()
    assert len(torch_keys) == len(gluon_keys)
    if reorder:
        key_words = ('running_mean', 'running_var', 'moving_mean', 'moving_var')
        torch_keys = [k for k in torch_keys if not k.endswith(key_words)] + \
                     [k for k in torch_keys if k.endswith(key_words)]
        gluon_keys = [k for k in gluon_keys if not k.endswith(key_words)] + \
                     [k for k in gluon_keys if k.endswith(key_words)]

    map = dict(zip(gluon_keys, torch_keys))
    pytorch_model_params = {}
    print('Convert Gluon Model to PyTorch Model ...')
    for key, value in gluon_model.collect_params().items():
        tensor = torch.from_numpy(value.data().asnumpy())
        tensor.require_grad = True
        pytorch_model_params[map[key]] = tensor
    torch.save(pytorch_model_params, os.path.join(torch_path, name + '.pth'))
    print('Finish')


if __name__ == '__main__':
    home = os.path.expanduser('~')

    parse = argparse.ArgumentParser(description='Convert gluon model to pytorch')
    parse.add_argument('--name', type=str, default='simple_pose_resnet50_v1d', help='name of the model')
    parse.add_argument('--gluon-path', type=str, default=os.path.join(home, '.mxnet/models'),
                       help='path to the gluon models')
    parse.add_argument('--torch-path', type=str, default=os.path.join(home, '.torch/models'),
                       help='path to the pytorch models')
    # for detection and segmentation
    parse.add_argument('--base', type=bool, default=False, help='use pretrained_base')
    parse.add_argument('--reorder', type=bool, default=False, help='reorder keys')  # for ssd

    config = parse.parse_args()
    gluon2torch(config.name, config.gluon_path, config.torch_path, config.base, config.reorder)
