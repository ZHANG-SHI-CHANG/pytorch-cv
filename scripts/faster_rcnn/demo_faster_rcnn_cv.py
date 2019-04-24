import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from data.transforms.rcnn_cv import load_test
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Test with Faster-RCNN networks.')
    parser.add_argument('--network', type=str, default='faster_rcnn_resnet50_v1b_voc',
                        help="Faster RCNN full network name")
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
                        help='Test demo images.')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='demo with GPU')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained model root.')
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
        device = torch.device('cuda')
    image = args.images
    net = get_model(args.network, pretrained=True, root=args.root)
    net.to(device)
    net.set_nms(0.3, 200)
    net.eval()

    ax = None
    x, img = load_test(image, short=net.short, max_size=net.max_size)
    x = x.to(device)

    with torch.no_grad():
        ids, scores, bboxes = [xx[0].cpu().numpy() for xx in net(x)]
    ax = plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                   class_names=net.classes, ax=ax)
    plt.show()
