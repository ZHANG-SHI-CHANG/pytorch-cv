import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils import data
from torch.backends import cudnn

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
import utils as ptutil
from model.model_zoo import get_model
from model.lr_scheduler_v2 import WarmupMultiStepLR, WarmupCosineLR
from data.helper import make_data_sampler, IterationBasedBatchSampler
from data.batchify import Tuple, Stack, Pad
from data.transforms.ssd_cv import SSDDefaultTrainTransform, SSDDefaultValTransform
from data.pascal_voc.detection_cv import VOCDetection
from data.mscoco.detection_cv import COCODetection
from utils.metrics import VOC07MApMetric, COCODetectionMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    # network
    parser.add_argument('--network', type=str, default='resnet50_v1s',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training mini-batch size')
    parser.add_argument('--test-batch-size', type=int, default=2,
                        help='Testing mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    # epoch, save and print config
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
    parser.add_argument('--log-step', type=int, default=1,
                        help='iteration to show results')
    parser.add_argument('--save-epoch', type=int, default=40,
                        help='epoch interval to save model.')
    parser.add_argument('--save-dir', type=str, default=cur_path,
                        help='Resume from previously saved parameters if not None.')
    parser.add_argument('--eval-epoch', type=int, default=40,
                        help='evaluating after epoch.')
    # lr
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',  # '160,200'
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--warmup-iters', type=int, default=500,  # 500
                        help='warmup epochs')
    parser.add_argument('--warmup-factor', type=float, default=1 / 3.0,
                        help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--lr-mode', type=str, default='cos',
                        help='learning mode')

    # device
    parser.add_argument('--cuda', type=ptutil.str2bool, default='true',
                        help='using CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    if args.lr == -1:
        args.lr = 1e-3 * args.batch_size / 32
    return args


def get_train_data(dataset):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    elif dataset.lower() == 'coco':
        train_dataset = COCODetection(splits='instances_train2017')
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset


def get_test_data(dataset):
    if dataset.lower() == 'voc':
        val_dataset = VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


class Trainer(object):
    def __init__(self, args):
        self.device = torch.device(args.device)
        net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
        self.save_prefix = net_name
        self.net = get_model(net_name, pretrained_base=True)
        if args.distributed:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        if args.resume.strip():
            logger.info("Resume from the model {}".format(args.resume))
            self.net.load_state_dict(torch.load(args.resume.strip()))
        else:
            logger.info("Init from base net {}".format(args.network))

        anchors = self.net.anchors()  # for dataset
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)  # for validate
        self.net.to(args.device)
        if args.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank], output_device=args.local_rank)

        # dataset and dataloader
        train_dataset = get_train_data(args.dataset)
        width, height = args.data_shape, args.data_shape
        batchify_fn = Tuple(Stack(), Stack(), Stack())
        train_dataset = train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors))
        args.per_iter = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iter = args.epochs * args.per_iter
        if args.distributed:
            sampler = data.DistributedSampler(train_dataset)
        else:
            sampler = data.RandomSampler(train_dataset)
        train_sampler = data.sampler.BatchSampler(sampler=sampler, batch_size=args.batch_size,
                                                  drop_last=True)
        train_sampler = IterationBasedBatchSampler(train_sampler, num_iterations=args.max_iter)
        self.train_loader = data.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True,
                                            collate_fn=batchify_fn, num_workers=args.num_workers)
        if args.eval_epoch > 0:
            # TODO: rewrite it
            val_dataset, self.metric = get_test_data(args.dataset)
            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            val_dataset = val_dataset.transform(SSDDefaultValTransform(width, height))
            val_sampler = make_data_sampler(val_dataset, False, args.distributed)
            val_batch_sampler = data.BatchSampler(val_sampler, args.test_batch_size, False)
            self.val_loader = data.DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                                              collate_fn=val_batchify_fn, num_workers=args.num_workers)

        # optimizer and lr scheduling
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.wd)
        # TODO: may need to change it
        if args.lr_mode == 'cos':
            self.scheduler = WarmupCosineLR(optimizer=self.optimizer, T_max=args.max_iter,
                                            warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters)
        elif args.lr_mode == 'step':
            lr_decay = float(args.lr_decay)
            milestones = sorted([float(ls) * args.per_iter for ls in args.lr_decay_epoch.split(',') if ls.strip()])
            self.scheduler = WarmupMultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=lr_decay,
                                               warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters)
        else:
            raise ValueError('illegal scheduler type')
        self.args = args

    def training(self):
        self.net.train()
        save_to_disk = ptutil.get_rank() == 0
        start_training_time = time.time()
        trained_time = 0
        tic = time.time()
        end = time.time()
        iteration, max_iter = 0, self.args.max_iter
        save_iter, eval_iter = self.args.per_iter * self.args.save_epoch, self.args.per_iter * self.args.eval_epoch
        # save_iter, eval_iter = self.args.per_iter * self.args.save_epoch, 10  # for debug
        logger.info("Start training, total epochs {:3d} = total iteration: {:6d}".format(self.args.epochs, max_iter))

        for i, batch in enumerate(self.train_loader):
            iteration += 1
            self.scheduler.step()
            image = batch[0].to(self.device)
            cls_targets = batch[1].to(self.device)
            box_targets = batch[2].to(self.device)

            self.optimizer.zero_grad()
            loss_dict = self.net(image, targets=(cls_targets, box_targets))
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = ptutil.reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()
            trained_time += time.time() - end
            end = time.time()
            if iteration % args.log_step == 0:
                eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
                log_str = ["Iteration {:06d} , Lr: {:.5f}, Cost: {:.2f}s, Eta: {}"
                               .format(iteration, self.optimizer.param_groups[0]['lr'], time.time() - tic,
                                       str(datetime.timedelta(seconds=eta_seconds))),
                           "total_loss: {:.3f}".format(losses_reduced.item())]
                for loss_name, loss_item in loss_dict_reduced.items():
                    log_str.append("{}: {:.3f}".format(loss_name, loss_item.item()))
                log_str = ', '.join(log_str)
                logger.info(log_str)
                tic = time.time()
            if save_to_disk and iteration % save_iter == 0:
                model_path = os.path.join(self.args.save_dir, "{}_iter_{:06d}.pth"
                                          .format(self.save_prefix, iteration))
                self.save_model(model_path)
            # Do eval when training, to trace the mAP changes and see performance improved whether or nor
            if args.eval_epoch > 0 and iteration % eval_iter == 0 and not iteration == max_iter:
                metrics = self.validate()
                ptutil.synchronize()
                names, values = ptutil.accumulate_metric(metrics)
                if names is not None:
                    log_str = ['{}: {:.5f}'.format(k, v) for k, v in zip(names, values)]
                    log_str = '\n'.join(log_str)
                    logger.info(log_str)
                self.net.train()
        if save_to_disk:
            model_path = os.path.join(self.args.save_dir, "{}_iter_{:06d}.pth"
                                      .format(self.save_prefix, max_iter))
            self.save_model(model_path)
        # compute training time
        total_training_time = int(time.time() - start_training_time)
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    def validate(self):
        self.metric.reset()
        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
            model = self.net.module
        else:
            model = self.net
        model.eval()
        tbar = tqdm(self.val_loader)
        for i, batch in enumerate(tbar):
            # if i == 5: break  # for debug
            image, label = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                ids, scores, bboxes = model(image)
            # clip to image size
            bboxes.clamp_(0, batch[0].shape[2])
            # split ground truths
            gt_ids = label.narrow(-1, 4, 1)
            gt_bboxes = label.narrow(-1, 0, 4)
            gt_difficults = label.narrow(-1, 5, 1) if label.shape[-1] > 5 else None

            self.metric.update(bboxes, ids, scores, gt_bboxes, gt_ids, gt_difficults)
        return self.metric

    def save_model(self, model_path):
        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
            model = self.net.module
        else:
            model = self.net
        torch.save(model.state_dict(), model_path)
        logger.info("Saved checkpoint to {}".format(model_path))


if __name__ == '__main__':
    args = parse_args()

    # device setting
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)
    args.lr = args.lr * args.num_gpus  # scale by num gpus

    logger = ptutil.setup_logger('SSD', args.save_dir, ptutil.get_rank(), 'log_ssd.txt', 'w')
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    trainer = Trainer(args)

    trainer.training()
    torch.cuda.empty_cache()
