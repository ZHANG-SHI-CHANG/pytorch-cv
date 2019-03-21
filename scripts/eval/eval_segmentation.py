import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data import get_segmentation_dataset
from data.base import make_data_sampler
from model.models_zoo.seg.segbase import MultiEvalModel
from utils.metrics.segmentation import SegmentationMetric
from utils.distributed.parallel import synchronize, is_main_process, accumulate_metric


def validate(evaluator, val_data, metric, device):
    tbar = tqdm(val_data)
    for i, (data, targets) in enumerate(tbar):
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            predicts = evaluator.forward(data)
        metric.update(targets, predicts)
    return metric


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Segmentation.')
    parser.add_argument('--model_name', type=str, default='fcn_resnet101_coco',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', action='store_true',
                        help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Select dataset.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

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
    model = model_zoo.get_model(args.model_name, pretrained=True, pretrained_base=False)
    model.to(device)

    # testing data
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    data_kwargs = {'transform': input_transform}

    val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', **data_kwargs)
    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    val_data = data.DataLoader(val_dataset, shuffle=False, batch_sampler=batch_sampler,
                               num_workers=args.num_workers)
    evaluator = MultiEvalModel(model, val_dataset.num_class)
    metric = SegmentationMetric(val_dataset.num_class)

    metric = validate(evaluator, val_data, metric, device)
    synchronize()
    pixAcc, mIoU = accumulate_metric(metric)
    if is_main_process():
        print('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
