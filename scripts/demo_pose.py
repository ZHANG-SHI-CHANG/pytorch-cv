import argparse
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn

from model.model_zoo import get_model
from data.transforms.yolo import load_test
from data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from utils.viz.keypoints import plot_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='Predict pose from a given image')
    parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                        help='name of the detection model to use')
    parser.add_argument('--pose-model', type=str, default='simple_pose_resnet50_v1b',
                        help='name of the pose estimation model to use')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='demo with GPU')
    parser.add_argument('--input-pic', type=str, default='./png/soccer.png',
                        help='path to the input picture')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cpu')
    if opt.cuda:
        cudnn.benchmark = True
        device = torch.device('cuda:0')

    detector = get_model(opt.detector, pretrained=True).to(device)
    detector.reset_class(["person"], reuse_weights=['person'])
    pose_net = get_model(opt.pose_model, pretrained=True).to(device)
    detector.eval()
    pose_net.eval()

    x, img = load_test(opt.input_pic, short=512)
    x = x.to(device)
    with torch.no_grad():
        class_IDs, scores, bounding_boxs = detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, device=device)
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                   box_thresh=0.5, keypoint_thresh=0.2)
    plt.show()
