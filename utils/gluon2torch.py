import os
import sys
import argparse
import torch
from gluoncv.model_zoo import get_model

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
from model import model_zoo


def gluon2torch(name, gluon_path, torch_path, base=False, reorder=False, force_pair=None):
    name = name.lower()
    if base:
        torch_model = model_zoo.get_model(name, pretrained=False, pretrained_base=False, root=torch_path)
        gluon_model = get_model(name, pretrained=True, pretrained_base=False, root=gluon_path)
    else:
        torch_model = model_zoo.get_model(name, pretrained=False, root=torch_path)
        gluon_model = get_model(name, pretrained=True, root=gluon_path)
    torch_keys = [k for k in torch_model.state_dict().keys() if not k.endswith('num_batches_tracked')]
    gluon_keys = gluon_model.collect_params().keys()
    if reorder:
        key_words = ('running_mean', 'running_var', 'moving_mean', 'moving_var')
        if force_pair is not None:
            torch_keys = [k for k in torch_keys if not k.endswith(key_words) and not k.startswith(force_pair[1])] + \
                         [k for k in torch_keys if not k.endswith(key_words) and k.startswith(force_pair[1])] + \
                         [k for k in torch_keys if k.endswith(key_words) and not k.startswith(force_pair[1])] + \
                         [k for k in torch_keys if k.endswith(key_words) and k.startswith(force_pair[1])]

            gluon_keys = [k for k in gluon_keys if not k.endswith(key_words) and not k.startswith(force_pair[0])] + \
                         [k for k in gluon_keys if not k.endswith(key_words) and k.startswith(force_pair[0])] + \
                         [k for k in gluon_keys if k.endswith(key_words) and not k.startswith(force_pair[0])] + \
                         [k for k in gluon_keys if k.endswith(key_words) and k.startswith(force_pair[0])]
        else:
            torch_keys = [k for k in torch_keys if not k.endswith(key_words)] + \
                         [k for k in torch_keys if k.endswith(key_words)]

            gluon_keys = [k for k in gluon_keys if not k.endswith(key_words)] + \
                         [k for k in gluon_keys if k.endswith(key_words)]

    assert len(torch_keys) == len(gluon_keys)
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
    parse.add_argument('--name', type=str, default='resnet101_v1s', help='name of the model')
    parse.add_argument('--gluon-path', type=str, default=os.path.join(home, '.mxnet/models'),
                       help='path to the gluon models')
    parse.add_argument('--torch-path', type=str, default=os.path.join(home, '.torch/models'),
                       help='path to the pytorch models')
    # for detection and segmentation
    parse.add_argument('--base', action='store_true', default=False, help='use pretrained_base')
    parse.add_argument('--reorder', action='store_true', default=False, help='reorder keys')  # for ssd
    parse.add_argument('--force-pair', type=str, default=None)
    # parse.add_argument('--force-pair', type=str, default='P,features.extra')

    config = parse.parse_args()
    if config.force_pair is not None:
        config.force_pair = tuple(config.force_pair.split(','))
    gluon2torch(config.name, config.gluon_path, config.torch_path,
                config.base, config.reorder, config.force_pair)
