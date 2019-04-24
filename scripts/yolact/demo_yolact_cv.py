import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from data.transforms.yolact_cv import load_test, post_process
from utils.viz.mask import plot_mask
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for YOLACT networks.')
    parser.add_argument('--network', type=str, default='yolact_fpn_resnet101_v1b_coco',
                        help="YOLACT full network name")
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/street.jpg'),
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--max-size', type=int, default=550,
                        help='Test images max size.')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='score threshold for observation.')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='demo with GPU')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained mdoel root.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda:0')
    image = args.images
    net = get_model(args.network, pretrained=True)
    net.to(device)
    net.eval()

    ax = None
    x, img = load_test(image, max_size=args.max_size)
    x = x.to(device)
    with torch.no_grad():
        out = net(x)[0]
        out = post_process(out, img.shape[1], img.shape[0], args.threshold)
        ids, scores, bboxes, masks = [a.cpu().numpy() for a in out]
    img = plot_mask(img, masks)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax = plot_bbox(img, bboxes, scores, ids,
                   class_names=net.classes, ax=ax, thresh=0.0)
    plt.show()
