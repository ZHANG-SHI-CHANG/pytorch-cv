import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from model.models_zoo.centernet import det_decode
from data.transforms.centernet_cv import load_demo, post_process
from utils.viz.bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Test with CenterNet networks.')
    parser.add_argument('--network', type=str, default='centernet_dla34_dcn_coco',
                        help="CenterNet full network name")
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
                        help='Test demo images.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='demo with GPU')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained model root.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.3,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--flip-test', action='store_true', default=False,
                        help='Using flipping test')
    parser.add_argument('--reg_offset', action='store_true', default=True,
                        help='Using regression offset')
    parser.add_argument('--topK', type=int, default=100, help='number of top K results')
    parser.add_argument('--scale', type=float, default=1.0, help='ratio scale')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')
    image = args.images
    net = get_model(args.network, pretrained=True, root=args.root)
    classes = net.classes
    num_classes = len(classes)

    net.to(device)
    net.eval()

    img, img_pre, meta = load_demo(image, scale=args.scale, flip_test=args.flip_test)
    img_pre = img_pre.to(device)

    with torch.no_grad():
        output = net(img_pre)
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if args.reg_offset else None
        if args.flip_test:
            hm = (hm[0:1] + torch.flip(hm[1:2], [3])) / 2
            wh = (wh[0:1] + torch.flip(wh[1:2], [3])) / 2
            reg = reg[0:1] if reg is not None else None
        dets = det_decode(hm, wh, reg=reg, K=args.topK)
    dets = dets.cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = post_process(dets.copy(), [meta['c']], [meta['s']],
                        meta['out_height'], meta['out_width'], num_classes)
    ids, scores, bboxes = list(), list(), list()
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= args.scale
        scores.append(dets[0][j][:, 4])
        bboxes.append(dets[0][j][:, :4])
        ids.append((np.array([j - 1] * dets[0][j].shape[0], dtype=np.float32)))
    ids = np.concatenate(ids, 0)
    scores = np.concatenate(scores, 0)
    bboxes = np.concatenate(bboxes, 0)
    ax = plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                   class_names=classes, reverse_rgb=True, ax=None)
    plt.show()
