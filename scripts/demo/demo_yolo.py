import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from data.transforms.yolo import load_test
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_mobilenet1.0_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='demo with GPU')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device('cuda:0')
    image_list = [os.path.join(cur_path, '../png/street.jpg')]
    net = get_model(args.network, pretrained=True)
    net.to(device)
    net.set_nms(0.45, 200)
    net.eval()

    for image in image_list:
        ax = None
        x, img = load_test(image, short=512)
        x = x.to(device)
        with torch.no_grad():
            ids, scores, bboxes = [xx[0].cpu().numpy() for xx in net(x)]
        ax = plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                       class_names=net.classes, ax=ax)
        plt.show()
