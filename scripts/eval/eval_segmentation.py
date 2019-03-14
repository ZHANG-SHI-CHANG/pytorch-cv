from tqdm import tqdm
import argparse

import torch
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms

from model import model_zoo
from data import get_segmentation_dataset
from model.models_zoo.seg.segbase import SegEvalModel
from utils.metrics.segmentation import SegmentationMetric


def validate(evaluator, val_data, metric, device):
    tbar = tqdm(val_data)
    for i, (data, targets) in enumerate(tbar):
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            predicts = evaluator.forward(data)
        metric.update(targets, predicts)
        pixAcc, mIoU = metric.get()
        tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Segmentation.')
    parser.add_argument('--model_name', type=str, default='fcn_resnet101_coco',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Training with GPUs.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Select dataset.')
    # for data
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
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
    model = model_zoo.get_model(args.model_name, pretrained=True, pretrained_base=False)

    # testing data
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}

    val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

    val_data = data.DataLoader(val_dataset, args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
    evaluator = SegEvalModel(model, device=device)
    metric = SegmentationMetric(val_dataset.num_class)

    validate(evaluator, val_data, metric, device)
