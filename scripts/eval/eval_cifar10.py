import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torch.utils import data
import torchvision.datasets as vdata
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.base import make_data_sampler
from utils.distributed.parallel import synchronize, accumulate_prediction, is_main_process
from utils.metrics.metric_classification import Accuracy


def get_dataloader(batch_size, num_workers, distributed):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_dataset = vdata.CIFAR10(root=os.path.expanduser('~/.torch/datasets/cifar10'),
                                train=False, transform=transform_test, download=True)

    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    return val_loader


def validate(net, val_data, device):
    net.to(device)
    net.eval()
    cpu_device = torch.device("cpu")
    results = list()
    tbar = tqdm(val_data)
    for i, (data, label) in enumerate(tbar):
        data = data.to(device)
        with torch.no_grad():
            outputs = net(data)
            outputs = outputs.to(cpu_device)
        results.append((label, outputs))
    return iter(results)


def parse_args():
    parser = argparse.ArgumentParser(description='Eval CIFAR10 networks.')
    parser.add_argument('--network', type=str, default='CIFAR_ResNet20_v1',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', action='store_true', help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args


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
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Load Model
    model_name = args.network
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        pretrained = True
    else:
        pretrained = False
    kwargs = {'classes': 10, 'pretrained': pretrained}
    net = model_zoo.get_model(model_name, **kwargs)

    # testing data
    val_metric = Accuracy()
    val_data = get_dataloader(args.batch_size, args.num_workers, distributed)

    # testing
    results = validate(net, val_data, device)
    synchronize()
    name, value = accumulate_prediction(results, val_metric)
    if is_main_process():
        print(name, value)
