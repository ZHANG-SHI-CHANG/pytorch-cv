# TODO: why Trainer object is slow ???
import os
import sys
import argparse
import warnings
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils import data
from torch.backends import cudnn

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
import utils as ptutil
from model.model_zoo import get_model
from data.base import make_data_sampler
from data.batchify import Tuple, Stack, Pad
from data.transforms.ssd_cv import SSDDefaultTrainTransform, SSDDefaultValTransform
from data.pascal_voc.detection_cv import VOCDetection
from data.mscoco.detection_cv import COCODetection
from utils.metrics import VOCMApMetric, COCODetectionMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training mini-batch size')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=2,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--first-valid', type=int, default=20,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')

    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = VOCDetection(splits=[(2007, 'test')])
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = COCODetection(splits='instances_train2017')
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


class Trainer(object):
    def __init__(self, args, device, distributed, logger):
        self.args = args
        self.best_map = 0.0
        self.device, self.distributed = device, distributed
        self.logger = logger
        net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
        self.args.save_prefix += net_name
        BatchNorm2d = torch.nn.SyncBatchNorm if distributed else torch.nn.BatchNorm2d
        self.net = get_model(net_name, pretrained_base=True, norm_layer=BatchNorm2d)
        if args.resume.strip():
            self.net.load_state_dict(torch.load(args.resume.strip()))
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
        # get property
        anchors = self.net.anchors()
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)  # for validate
        self.net.to(device)
        if distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank], output_device=args.local_rank)

        # dataset and dataloader
        train_dataset, val_dataset, self.metric = get_dataset(args.dataset, args)
        width, height = args.data_shape, args.data_shape
        batchify_fn = Tuple(Stack(), Stack(), Stack())
        train_dataset = train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors))
        train_sampler = make_data_sampler(train_dataset, True, distributed)
        train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, True)
        self.train_loader = data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                            collate_fn=batchify_fn, num_workers=args.num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_dataset = val_dataset.transform(SSDDefaultValTransform(width, height))
        val_sampler = make_data_sampler(val_dataset, False, distributed)
        val_batch_sampler = data.BatchSampler(val_sampler, args.test_batch_size, False)
        self.val_loader = data.DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                                          collate_fn=val_batchify_fn, num_workers=args.num_workers)

        # optimizer and lr scheduling
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.wd)
        lr_decay = float(args.lr_decay)
        lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, lr_decay)

    def training(self, epoch):
        self.net.train()
        tbar = tqdm(self.train_loader)
        self.lr_scheduler.step()
        regs_losses, cls_losses = 0.0, 0.0

        # import numpy as np
        # np.random.seed(10)
        # image = np.random.randn(1, 3, 300, 300).astype(np.float32)
        # cls_target = np.random.randint(0, 20, (1, 8732)).astype(np.float32)
        # box_target = np.random.randn(1, 8732, 4).astype(np.float32)
        #
        # image = torch.from_numpy(image)
        # cls_target = torch.from_numpy(cls_target)
        # box_target = torch.from_numpy(box_target)
        # regs_loss, cls_loss = self.net(image, targets=(cls_target, box_target))
        # print(regs_loss, cls_loss)

        for i, batch in enumerate(tbar):
            # if i == 10: break
            image = batch[0].to(self.device)
            cls_targets = batch[1].to(self.device)
            box_targets = batch[2].to(self.device)
            regs_loss, cls_loss = self.net(image, targets=(cls_targets, box_targets))
            sum_loss = regs_loss + cls_loss
            self.optimizer.zero_grad()
            sum_loss.backward()
            self.optimizer.step()
            regs_losses += regs_loss.item()
            cls_losses += cls_loss.item()
            if self.args.log_interval and not (i + 1) % self.args.log_interval and ptutil.is_main_process():
                self.logger.info('[Epoch {}][Batch {}], regression loss={:.3f}, classification loss={:.3f}'
                                 .format(epoch, i, regs_losses / (i + 1), cls_losses / (i + 1)))

    def validate(self):
        self.metric.reset()
        self.net.eval()
        tbar = tqdm(self.val_loader)
        for i, batch in enumerate(tbar):
            # if i == 10: break
            image, label = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                ids, scores, bboxes = self.net(image)
            det_ids = [ids]
            det_scores = [scores]
            # clip to image size
            det_bboxes = [bboxes.clamp(0, batch[0].shape[2])]
            # split ground truths
            gt_ids = [label.narrow(-1, 4, 1)]
            gt_bboxes = [label.narrow(-1, 0, 4)]
            gt_difficults = [label.narrow(-1, 5, 1) if label.shape[-1] > 5 else None]

            self.metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return self.metric

    def save_params(self, current_map, epoch):
        current_map = float(current_map)
        if current_map > self.best_map:
            self.best_map = current_map
            filename = '{:s}_best.params'.format(self.args.save_prefix, epoch, current_map)
            torch.save(self.net.module.state_dict() if self.distributed else self.net.state_dict(), filename)
        if self.args.save_interval and epoch % self.args.save_interval == 0:
            filename = '{:s}_{:04d}_{:.4f}.params'.format(self.args.save_prefix, epoch, current_map)
            torch.save(self.net.module.state_dict() if self.distributed else self.net.state_dict(), filename)


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cpu')
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        distributed = False

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)
        ptutil.synchronize()

    logger = ptutil.setup_logger('SSD', cur_path, ptutil.get_rank(), 'log_ssd.txt', 'w')

    logger.info(args)
    trainer = Trainer(args, device, distributed, logger)

    logger.info('Starting Epoch: {}'.format(args.start_epoch))
    logger.info('Total Epochs: {}'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch > args.first_valid and not (epoch + 1) % args.val_interval:
            metric = trainer.validate()
            ptutil.synchronize()
            map_name, mean_ap = ptutil.accumulate_metric(metric)
            if ptutil.is_main_process():
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        if ptutil.is_main_process():
            trainer.save_params(current_map, epoch)
