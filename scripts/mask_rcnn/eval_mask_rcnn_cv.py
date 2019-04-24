# TODO: not finish
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils import data

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from model.ops import bbox_clip_to_image
from data.helper import make_data_sampler
from data.batchify import Tuple, Append
from data.transforms.mask import fill
from data.mscoco.instance_cv import COCOInstance
from data.transforms.rcnn_cv import MaskRCNNDefaultValTransform
from utils.metrics.coco_instance import COCOInstanceMetric
from utils.distributed.parallel import synchronize, accumulate_metric, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Mask-RCNN networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=1,  # now, only support one
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained model root')
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
    # device
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def get_dataset(short, max_size, args):
    transform = MaskRCNNDefaultValTransform(short, max_size)
    if args.dataset.lower() == 'coco':
        val_dataset = COCOInstance(splits='instances_val2017', skip_empty=False, transform=transform)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval',
                                        cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(args.dataset))
    return val_dataset, val_metric


def get_dataloader(val_dataset, batch_size, num_workers, distributed):
    """Get dataloader."""
    batchify_fn = Tuple(*[Append() for _ in range(2)])
    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn,
                                 num_workers=num_workers)
    return val_loader


def validate(net, val_data, device, metric):
    metric.reset()
    net.eval()
    tbar = tqdm(val_data)

    for ib, batch in enumerate(tbar):
        # if ib == 5: break
        x = batch[0][0].to(device)
        im_info = batch[1][0].numpy()
        with torch.no_grad():
            ids, scores, bboxes, masks = net(x)
            bboxes = bbox_clip_to_image(bboxes, x).cpu().numpy()
            ids, scores, masks = ids.cpu().numpy(), scores.cpu().numpy(), masks.cpu().numpy()
        im_height, im_width, im_scale = im_info.squeeze()
        valid = np.where(((ids >= 0) & (scores >= 0.001)))[0]
        ids = ids[valid]
        scores = scores[valid]
        bboxes = bboxes[valid] / im_scale
        masks = masks[valid]
        im_height, im_width = int(round(im_height / im_scale)), int(round(im_width / im_scale))
        full_masks = []
        for bbox, mask in zip(bboxes, masks):
            full_masks.append(fill(mask, bbox, (im_width, im_height)))
        full_masks = np.array(full_masks)
        metric.update(bboxes, ids, scores, full_masks)

    return metric.get()


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
    net_name = '_'.join(('mask_rcnn', *module_list, args.network, args.dataset))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = model_zoo.get_model(net_name, pretrained=True, keep_max=True, root=args.root, **kwargs)
    else:
        net = model_zoo.get_model(net_name, pretrained=False, **kwargs)
        net.load_state_dict(args.pretrained.strip())

    net.to(device)

    # testing data
    val_dataset, val_metric = get_dataset(net.short, net.max_size, args)
    val_data = get_dataloader(val_dataset, args.batch_size, args.num_workers,
                              distributed)
    classes = val_dataset.classes  # class names

    names, values = validate(net, val_data, device, val_metric)
    for k, v in zip(names, values):
        print(k, v)

    # testing
    # val_metric = validate(net, val_data, device, val_metric)
    # synchronize()
    # names, values = accumulate_metric(val_metric)
    # if is_main_process():
    #     for k, v in zip(names, values):
    #         print(k, v)
