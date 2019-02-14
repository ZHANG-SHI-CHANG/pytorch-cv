import argparse

import torch
import matplotlib.pyplot as plt

from model.model_zoo import get_model
from data.transforms.ssd import load_test
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Test with SSD networks.')
    parser.add_argument('--network', type=str, default='ssd_512_mobilenet1.0_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    image_list = ['png/street.jpg']
    net = get_model(args.network, pretrained=True)
    net.set_nms(0.45, 200)
    net.eval()

    ax = None
    for image in image_list:
        x, img = load_test(image, short=512)
        with torch.no_grad():
            ids, scores, bboxes = [xx[0].numpy() for xx in net(x)]
        ax = plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                       class_names=net.classes, ax=ax)
        plt.show()
