import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torch.utils import data

from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.helper import make_data_sampler
from data.cifar10 import CIFAR10
from data.transforms.utils import transforms_cv
from utils.distributed.parallel import synchronize, accumulate_metric, is_main_process
from utils.metrics.classification_pt import Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Eval CIFAR10 networks.')
    parser.add_argument('--network', type=str, default='CIFAR_ResNet20_v1',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='pre-trained model root')
    parser.add_argument('--data-root', type=str, default=os.path.expanduser('~/.torch/datasets/cifar10'),
                        help='dataset root')
    # device
    parser.add_argument('--cuda', action='store_true', default=True, help='Evaluate with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def get_dataloader(batch_size, num_workers, data_root, distributed):
    transform_test = transforms.Compose([
        transforms_cv.ToTensor(),
        transforms_cv.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_dataset = CIFAR10(root=data_root, train=False, transform=transform_test, download=True)

    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    return val_loader


def validate(net, val_data, device, metric):
    net.eval()
    cpu_device = torch.device("cpu")
    tbar = tqdm(val_data)
    for i, (data, label) in enumerate(tbar):
        data = data.to(device)
        with torch.no_grad():
            outputs = net(data)
        outputs = outputs.to(cpu_device)
        metric.update(label, outputs)
    return metric


if __name__ == '__main__':
    args = parse_args()

    # device
    device = torch.device('cpu')
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if args.cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        distributed = False

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)

    # Load Model
    model_name = args.network
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        pretrained = True
    else:
        pretrained = False
    kwargs = {'classes': 10, 'pretrained': pretrained, 'root': args.root}
    net = model_zoo.get_model(model_name, **kwargs)
    net.to(device)

    # testing data
    val_metric = Accuracy()
    val_data = get_dataloader(args.batch_size, args.num_workers, args.data_root, distributed)

    # testing
    metric = validate(net, val_data, device, val_metric)
    synchronize()
    name, value = accumulate_metric(metric)
    if is_main_process():
        print(name, value)
