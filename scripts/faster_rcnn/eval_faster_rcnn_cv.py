from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch.utils import data

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from model.ops import bbox_clip_to_image
from data.helper import make_data_sampler
from data.batchify import Tuple, Empty, Append
from data.pascal_voc.detection_cv import VOCDetection
from data.transforms.rcnn_cv import FasterRCNNDefaultValTransform
from data.mscoco.detection_cv import COCODetection
from utils.metrics.voc_detection_pt import VOC07MApMetric
from utils.metrics.coco_detection import COCODetectionMetric
from utils.distributed.parallel import synchronize, accumulate_metric, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Faster-RCNN networks.')
    parser.add_argument('--network', type=str, default='resnet101_v1d',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=1,  # now, only support one
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')
    parser.add_argument('--use-fpn', action='store_true', default=False,
                        help='Whether to use feature pyramid network.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained model root')
    # device
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Training with GPUs')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def get_dataset(short, max_size, dataset):
    transform = FasterRCNNDefaultValTransform(short, max_size)
    if dataset.lower() == 'voc':
        val_dataset = VOCDetection(splits=[(2007, 'test')], transform=transform)
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False,
                                    transform=transform, keep_idx=True)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(val_dataset, batch_size, num_workers, distributed, coco=False):
    """Get dataloader."""
    if coco:
        batchify_fn = Tuple(*[Append() for _ in range(3)], Empty())
    else:
        batchify_fn = Tuple(*[Append() for _ in range(3)])
    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn,
                                 num_workers=num_workers)
    return val_loader


def validate(net, val_data, device, metric, coco=False):
    metric.reset()
    net.eval()
    tbar = tqdm(val_data)

    for ib, batch in enumerate(tbar):
        x = batch[0][0].to(device)
        y = batch[1][0].to(device)
        im_scale = batch[2][0]
        with torch.no_grad():
            ids, scores, bboxes = net(x)
        det_ids = [ids]
        det_scores = [scores]
        # clip to image size
        det_bboxes = [bbox_clip_to_image(bboxes, x)]
        # rescale to original resolution
        im_scale = im_scale.reshape((-1)).item()
        det_bboxes[-1] *= im_scale
        # split ground truths
        gt_ids = [y.narrow(-1, 4, 1)]
        gt_bboxes = [y.narrow(-1, 0, 4)]
        gt_bboxes[-1] *= im_scale
        gt_difficults = [y.narrow(-1, 5, 1) if y.shape[-1] > 5 else None]

        if coco:
            metric.update(det_bboxes, det_ids, det_scores, batch[3], gt_bboxes, gt_ids, gt_difficults)
        else:
            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return metric


if __name__ == '__main__':
    args = parse_args()

    # device
    device = torch.device('cpu')
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        distributed = False

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
    net_name = '_'.join(('faster_rcnn', *module_list, args.network, args.dataset))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = model_zoo.get_model(net_name, pretrained=True, root=args.root, **kwargs)
    else:
        net = model_zoo.get_model(net_name, pretrained=False, **kwargs)
        net.load_state_dict(args.pretrained.strip())

    net.to(device)

    # testing data
    val_dataset, val_metric = get_dataset(net.short, net.max_size, args.dataset)
    val_data = get_dataloader(val_dataset, args.batch_size, args.num_workers,
                              distributed, args.dataset == 'coco')
    classes = val_dataset.classes  # class names

    # testing
    val_metric = validate(net, val_data, device, val_metric, args.dataset == 'coco')
    synchronize()
    names, values = accumulate_metric(val_metric)
    if is_main_process():
        for k, v in zip(names, values):
            print(k, v)
