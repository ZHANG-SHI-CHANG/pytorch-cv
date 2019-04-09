import os
import sys
import argparse
import math
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.base import make_data_sampler
from data.imagenet.classification import ImageNet
from utils.distributed.parallel import synchronize, accumulate_metric, is_main_process
from utils.metrics.classification_pt import Accuracy, TopKAccuracy


def get_dataloader(opt, distributed):
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = ImageNet(opt.data_dir, train=False, transform=transform_test)

    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=opt.batch_size, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_sampler=batch_sampler, num_workers=opt.num_workers)
    return val_loader


def validate(net, val_data, device, acc_top1, acc_top5):
    net.eval()
    acc_top1.reset()
    acc_top5.reset()
    # cpu_device = torch.device("cpu")
    tbar = tqdm(val_data)
    for i, (data, label) in enumerate(tbar):
        data, label = data.to(device), label.to(device)
        data = data.to(device)
        with torch.no_grad():
            outputs = net(data)
        # outputs = outputs.to(cpu_device)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
    return acc_top1, acc_top5


def parse_args():
    parser = argparse.ArgumentParser(description='Eval ImageNet networks.')
    parser.add_argument('--model', type=str, default='resnet18_v1b_0.89',
                        help="Base network name")
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Evaluate with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/.torch/datasets/imagenet'),
                        help='default data root')
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
        synchronize()

    # Load Model
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        pretrained = True
    else:
        pretrained = False
    model_name = args.model
    kwargs = {'classes': 1000, 'pretrained': pretrained}

    net = model_zoo.get_model(model_name, **kwargs)
    net.to(device)

    # testing data
    acc_top1 = Accuracy()
    acc_top5 = TopKAccuracy(5)
    val_data = get_dataloader(args, distributed)

    # testing
    acc_top1, acc_top5 = validate(net, val_data, device, acc_top1, acc_top5)
    synchronize()
    name1, top1 = accumulate_metric(acc_top1)
    name5, top5 = accumulate_metric(acc_top5)
    if is_main_process():
        print('%s: %f, %s: %f' % (name1, top1, name5, top5))
