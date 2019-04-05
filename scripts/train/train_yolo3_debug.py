import os
import sys
import argparse
import warnings
from tqdm import tqdm
import numpy as np

import torch
from torch import optim
from torch.utils import data
from torch.backends import cudnn

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
import utils as ptutil
from data.base import make_data_sampler
from data.randomloader import RandomTransformDataLoader
from data.batchify import Tuple, Stack, Pad
from data.pascal_voc.detection_cv import VOCDetection
from data.mscoco.detection_cv import COCODetection
from data.transforms.yolo_cv import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform
from utils.metrics import VOC07MApMetric, COCODetectionMetric
from model.model_zoo import get_model
from model.lr_scheduler import LRScheduler


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:
        from data.mixup.detection import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


class Trainer(object):
    def __init__(self, args, device, distributed, logger):
        self.args = args
        self.device, self.distributed = device, distributed
        self.logger = logger
        self.best_map = 0.
        # create network
        net_name = '_'.join(('yolo3', args.network, args.dataset))
        self.args.save_prefix += net_name
        BatchNorm2d = torch.nn.SyncBatchNorm if distributed else torch.nn.BatchNorm2d

        self.net = get_model(net_name, pretrained_base=True, norm_layer=BatchNorm2d)
        self.net.to(device)
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        if args.resume.strip():
            self.net.load_state_dict(torch.load(args.resume.strip()))
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # self.net.initialize()  # TODO

        # get property
        classes, anchors = self.net.num_class, self.net.anchors
        if args.label_smooth:
            self.net._target_generator._label_smooth = True

        # TODO: have bug
        print('before distributed')
        if distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank], output_device=args.local_rank)
        print('after distributed')

        # dataset and dataloader
        train_dataset, val_dataset, self.metric = get_dataset(args.dataset, args)
        # pre-config
        width, height = args.data_shape, args.data_shape
        batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
        train_dataset = train_dataset.transform(
            YOLO3DefaultTrainTransform(width, height, classes, anchors, mixup=args.mixup))
        train_sampler = make_data_sampler(train_dataset, True, distributed)
        train_batch_sampler = data.sampler.BatchSampler(train_sampler, args.batch_size, True)
        if args.no_random_shape:
            self.train_loader = data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                collate_fn=batchify_fn, num_workers=args.num_workers)
        else:
            transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, classes, anchors, mixup=args.mixup)
                             for x in range(10, 20)]
            self.train_loader = RandomTransformDataLoader(transform_fns, train_dataset,
                                                          batch_sampler=train_batch_sampler,
                                                          collate_fn=batchify_fn, num_workers=args.num_workers)

        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_dataset = val_dataset.transform(YOLO3DefaultValTransform(width, height))
        val_sampler = make_data_sampler(val_dataset, False, distributed)
        val_batch_sampler = data.sampler.BatchSampler(val_sampler, args.test_batch_size, False)
        self.val_loader = data.DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                                          collate_fn=val_batchify_fn, num_workers=args.num_workers)
        # optimizer and lr scheduling
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.wd)
        if args.lr_decay_period > 0:
            lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
        self.lr_scheduler = LRScheduler(self.optimizer, mode=args.lr_mode,
                                        n_iters=len(self.train_loader),
                                        n_epochs=args.epochs, n_step=lr_decay_epoch,
                                        step_factor=args.lr_decay, power=2,
                                        warmup_epochs=args.warmup_epochs)

    def training(self, epoch):
        self.net.train()
        if args.mixup:
            self.train_loader.dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                self.train_loader.dataset._data.set_mixup(None)
        tbar = tqdm(self.train_loader)
        obj_losses, center_losses, scale_losses, cls_losses = 0., 0., 0., 0.
        for i, batch in enumerate(tbar):
            if i == 10:
                break
            image = batch[0].to(self.device)
            fixed_targets = [batch[it].to(self.device) for it in range(1, 6)]
            gt_boxes = batch[6].to(self.device)
            obj_loss, center_loss, scale_loss, cls_loss = self.net(image, gt_boxes, *fixed_targets)
            sum_loss = obj_loss + center_loss + scale_loss + cls_loss
            self.lr_scheduler.step(i, epoch)
            self.optimizer.zero_grad()
            sum_loss.backward()
            self.optimizer.step()
            obj_losses += obj_loss.item()
            center_losses += center_loss.item()
            scale_losses += scale_loss.item()
            cls_losses += cls_loss.item()
            if self.args.log_interval and not (i + 1) % self.args.log_interval and ptutil.is_main_process():
                self.logger.info('[Epoch {}][Batch {}], obj_loss={:.3f}, center_loss={:.3f}, scale_loss={:.3f}, '
                                 'cls_loss={:.3f}'.format(epoch, i, obj_losses / (i + 1), center_losses / (i + 1),
                                                          scale_losses / (i + 1), cls_losses / (i + 1)))

    def validate(self):
        self.metric.reset()
        self.net.eval()
        tbar = tqdm(self.val_loader)
        for i, batch in enumerate(tbar):
            if i == 10:
                break
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training mini-batch size')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epochs at which learning rate decays. default is 160,180.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=1,  # 100
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--no-random-shape', action='store_true', default=False,
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                             'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    args = parser.parse_args()
    return args


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

    logger = ptutil.setup_logger('YOLO', cur_path, ptutil.get_rank(), 'log_yolo.txt', 'w')
    logger.info(args)

    trainer = Trainer(args, device, distributed, logger)

    logger.info('Starting Epoch: {}'.format(args.start_epoch))
    logger.info('Total Epochs: {}'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        # trainer.training(epoch)
        if not (epoch + 1) % args.val_interval:
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
