import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib

import torch
from torch.backends import cudnn
from torch import optim
from torch.utils import data
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10

matplotlib.use('Agg')
cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.base import make_data_sampler
from utils.filesystem import makedirs
from utils.plot_history import TrainingHistory
from utils.distributed.parallel import synchronize, all_gather, is_main_process, reduce_list, accumulate_metric
from utils.metrics.metric_classification import Accuracy
from utils.optim_utils import get_learning_rate, set_learning_rate


class Solver(object):
    def __init__(self, net, cfg, device, distributed):
        super(Solver, self).__init__()
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.net = net
        self.cfg, self.distributed = cfg, distributed
        self.device = device

        self.lr_decay_epoch = [int(i) for i in cfg.lr_decay_epoch.split(',')] + [np.inf]
        self.save_period = args.save_period
        if cfg.save_dir and self.save_period:
            self.save_dir = cfg.save_dir
            makedirs(self.save_dir)

        self.plot_path = cfg.save_plot_dir

        if is_main_process():
            logging.basicConfig(level=logging.INFO)
            logging.info(args)

    def train(self):
        train_dataset = CIFAR10(root=os.path.expanduser('~/.torch/datasets/cifar10'),
                                train=True, transform=self.transform_train, download=True)
        train_sampler = make_data_sampler(train_dataset, True, self.distributed)
        train_batch_sampler = data.sampler.BatchSampler(train_sampler, self.cfg.batch_size, True)
        train_data = data.DataLoader(train_dataset, num_workers=self.cfg.num_workers,
                                     batch_sampler=train_batch_sampler)

        val_dataset = CIFAR10(root=os.path.expanduser('~/.torch/datasets/cifar10'),
                              train=False, transform=self.transform_test)
        val_sampler = make_data_sampler(val_dataset, False, self.distributed)
        val_batch_sampler = data.sampler.BatchSampler(val_sampler, self.cfg.batch_size, False)
        val_data = data.DataLoader(val_dataset, num_workers=self.cfg.num_workers,
                                   batch_sampler=val_batch_sampler)

        optimizer = optim.SGD(self.net.parameters(), nesterov=True, lr=self.cfg.lr, weight_decay=self.cfg.wd,
                              momentum=self.cfg.momentum)
        metric = Accuracy()
        train_metric = Accuracy()
        loss_fn = nn.CrossEntropyLoss()
        if is_main_process():
            train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0
        best_val_score = 0

        for epoch in range(self.cfg.num_epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)

            if epoch == self.lr_decay_epoch[lr_decay_count]:
                set_learning_rate(optimizer, get_learning_rate(optimizer) * self.cfg.lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                image = batch[0].to(self.device)
                label = batch[1].to(self.device)

                output = self.net(image)
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_metric.update(label, output)
                iteration += 1

            metric = self.validate(val_data, metric)
            synchronize()
            train_loss /= num_batch
            train_loss = reduce_list(all_gather(train_loss))
            name, acc = accumulate_metric(train_metric)
            name, val_acc = accumulate_metric(metric)
            if is_main_process():
                train_history.update([1 - acc, 1 - val_acc])
                train_history.plot(save_path='%s/%s_history.png' % (self.plot_path, self.cfg.model))
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    torch.save(self.net.state_dict(), '%s/%.4f-cifar-%s-%d-best.pth' %
                               (self.save_dir, best_val_score, self.cfg.model, epoch))
                logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                             (epoch, acc, val_acc, train_loss, time.time() - tic))

                if self.save_period and self.cfg.save_dir and (epoch + 1) % self.save_period == 0:
                    torch.save(self.net.state_dict(), '%s/cifar10-%s-%d.pth' % (self.save_dir, self.cfg.model, epoch))

        if is_main_process() and self.save_period and self.save_dir:
            torch.save(self.net.state_dict(), '%s/cifar10-%s-%d.pth'
                       % (self.save_dir, self.cfg.model, self.cfg.num_epochs - 1))

    def validate(self, val_data, metric):
        self.net.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_data):
                image = batch[0].to(self.device)
                label = batch[1].to(self.device)
                outputs = self.net(image)
                metric.update(label, outputs)
        self.net.train()
        return metric


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--cuda', action='store_true', help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model', type=str, default='cifar_resnet20_v1',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default=os.path.join(cur_path, 'params'),
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default=cur_path,
                        help='the path to save the history plot')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_args()
    classes = 10

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

    BatchNorm2d = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
    model_name = args.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                  'drop_rate': args.drop_rate}
    else:
        kwargs = {'classes': classes}

    net = model_zoo.get_model(model_name, norm_layer=BatchNorm2d, **kwargs)
    net = net.to(device)

    if args.resume_from:
        net.load_state_dict(torch.load(args.resume_from))
    if distributed:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], output_device=args.local_rank)

    solver = Solver(net, args, device, distributed)
    solver.train()
