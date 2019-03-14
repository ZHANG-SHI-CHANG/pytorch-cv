from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch.backends import cudnn
from torch.utils import data

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.batchify import Tuple, Stack, Pad
from data.pascal_voc.detection import VOCDetection
from data.mscoco.detection import COCODetection
from utils.metrics.voc_detection import VOC07MApMetric
from utils.metrics.coco_detection import COCODetectionMetric
from data.transforms.ssd import SSDDefaultValTransform


def get_dataset(dataset, data_shape):
    transform = SSDDefaultValTransform(data_shape, data_shape)
    if dataset.lower() == 'voc':
        val_dataset = VOCDetection(splits=[(2007, 'test')], transform=transform)
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False, transform=transform)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(val_dataset, batch_size, num_workers):
    """Get dataloader."""
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = data.DataLoader(val_dataset, collate_fn=batchify_fn,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader


# TODO: support multiple gpu
def validate(net, val_data, device, size, metric):
    net = net.to(device)
    metric.reset()
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.eval()
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data):
            data = batch[0].to(device)
            label = batch[1].to(device)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            x, y = data, label
            with torch.no_grad():
                ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clamp(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.narrow(-1, 4, 1))
            gt_bboxes.append(y.narrow(-1, 0, 4))
            gt_difficults.append(y.narrow(-1, 5, 1) if y.shape[-1] > 5 else None)

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])
    return metric.get()


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
    parser.add_argument('--cuda', action='store_true', help='Training with GPUs.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    device = torch.device('cpu')
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device('cuda:0')

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = model_zoo.get_model(net_name, pretrained=True)
    else:
        net = model_zoo.get_model(net_name, pretrained=False)
        net.load_parameters(args.pretrained.strip())

    # training data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    val_data = get_dataloader(val_dataset, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    # training
    names, values = validate(net, val_data, device, len(val_dataset), val_metric)
    for k, v in zip(names, values):
        print(k, v)
