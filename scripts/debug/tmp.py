import os
import sys
import argparse

import torch
from torch import nn
from torch.backends import cudnn

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
import utils as ptutil


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.features = nn.ModuleList()
        self.features.append(nn.Conv2d(3, 10, 1, 1, 1))
        for i in range(10):
            self.features.append(nn.Conv2d(10, 10, 1, 1, 1))

    def forward(self, x):
        for feat in self.features:
            x = feat(x)
        return x


class Net(nn.Module):
    def __init__(self, stage, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(Net, self).__init__()
        self.stage = nn.ModuleList(stage)
        self.l1 = nn.Conv2d(3, 10, 1, 1, 1)
        self.l2 = norm_layer(10, **({} if norm_kwargs is None else norm_kwargs))
        self.l3 = nn.ModuleList([nn.Conv2d(10, 10, 1, 1, 1),
                                 nn.ReLU(inplace=True)])

    def forward(self, x):
        return self.l2(self.l1(x))


def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--network', type=str, default='mobilenet1.0',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
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
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)
        ptutil.synchronize()

    BatchNorm2d = nn.SyncBatchNorm if distributed else nn.BatchNorm2d

    base = Stage()
    stage = [base.features[:2], base.features[2:4]]
    net = Net(stage, norm_layer=BatchNorm2d)
    net.to(device)

    print('before distributed')
    if distributed:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], output_device=args.local_rank)

    print('after distributed')
