import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from data.transforms.rcnn_cv import load_test
from utils.viz.mask import plot_mask, expand_mask
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Mask-RCNN networks.')
    parser.add_argument('--network', type=str, default='mask_rcnn_resnet50_v1b_coco',
                        help="Mask RCNN full network name")
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
                        help='Test images, use comma to split multiple.')
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
        device = torch.device('cuda')
    image = args.images
    net = get_model(args.network, pretrained=True)
    net.to(device)
    net.set_nms(0.3, 200)
    net.eval()

    ax = None
    x, img = load_test(image, short=net.short, max_size=net.max_size)
    x = x.to(device)
    with torch.no_grad():
        ids, scores, bboxes, masks = [xx.cpu().numpy() for xx in net(x)]
    masks = expand_mask(masks, bboxes, (img.shape[1], img.shape[0]), scores)
    img = plot_mask(img, masks)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax = plot_bbox(img, bboxes, scores, ids,
                   class_names=net.classes, ax=ax)
    plt.show()
