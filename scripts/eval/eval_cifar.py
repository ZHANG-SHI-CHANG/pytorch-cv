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
from utils.metrics.metric_classification import Accuracy


def get_dataloader(batch_size, num_workers):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_dataset = vdata.CIFAR10(root=os.path.expanduser('~/.torch/datasets/cifar10'),
                                train=False, transform=transform_test, download=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    return val_loader


def validate(net, val_data, device, metric):
    net = net.to(device)
    net.eval()
    tbar = tqdm(val_data)
    for i, (data, label) in enumerate(tbar):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            outputs = net(data)
        metric.update(label, outputs)
    return metric.get()


def parse_args():
    parser = argparse.ArgumentParser(description='Eval CIFAR10 networks.')
    parser.add_argument('--network', type=str, default='CIFAR_ResNeXt29_16x64d',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Training with GPUs.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    device = torch.device('cpu')
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device('cuda:0')

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
    val_data = get_dataloader(args.batch_size, args.num_workers)

    # testing
    names, values = validate(net, val_data, device, val_metric)
    print(names, values)


