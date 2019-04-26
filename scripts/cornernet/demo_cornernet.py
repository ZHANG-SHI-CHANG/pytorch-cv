# TODO unfinish
# import os
# import sys
# import argparse
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#
# cur_path = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(cur_path, '../..'))
# from model.model_zoo import get_model
# from utils.viz.bbox import plot_bbox
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Demo with CornerNet networks.')
#     parser.add_argument('--network', type=str, default='corner_squeeze_hourglass_coco',
#                         help="CenterNet full network name")
#     parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
#                         help='Test demo images.')
#     parser.add_argument('--cuda', action='store_true', default=True,
#                         help='demo with GPU')
#     parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
#                         help='Default pre-trained model root.')
#     parser.add_argument('--pretrained', type=str, default='True',
#                         help='Load weights from previously saved parameters.')
#     parser.add_argument('--thresh', type=float, default=0.3,
#                         help='Threshold of object score when visualize the bboxes.')
#     parser.add_argument('--flip-test', action='store_true', default=False,
#                         help='Using flipping test')
#     parser.add_argument('--reg_offset', action='store_true', default=True,
#                         help='Using regression offset')
#     parser.add_argument('--topK', type=int, default=100, help='number of top K results')
#     parser.add_argument('--scale', type=float, default=1.0, help='ratio scale')
#
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     device = torch.device('cpu')
#     if args.cuda:
#         device = torch.device('cuda')
#
#     image = cv2.imread(args.images)
#     net = get_model(args.network, pretrained=True, root=args.root)
#
