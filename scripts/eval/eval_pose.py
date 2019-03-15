from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch.backends import cudnn
from torch.utils import data

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model import model_zoo
from data.mscoco.keypoints import COCOKeyPoints
from data.transforms.pose import flip_heatmap, get_final_preds
from data.transforms.simple_pose import SimplePoseDefaultValTransform
from utils.metrics.coco_keypoints import COCOKeyPointsMetric


def get_dataloader(data_dir, batch_size, num_workers, input_size, mean, std):
    """Get dataloader."""

    def val_batch_fn(batch, device):
        data = [batch[0].to(device)]
        scale = batch[1]
        center = batch[2]
        score = batch[3]
        imgid = batch[4]
        return data, scale, center, score, imgid

    val_dataset = COCOKeyPoints(data_dir, aspect_ratio=4. / 3.,
                                splits=('person_keypoints_val2017'))

    meanvec = [float(i) for i in mean.split(',')]
    stdvec = [float(i) for i in std.split(',')]
    transform_val = SimplePoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                  joint_pairs=val_dataset.joint_pairs,
                                                  image_size=input_size,
                                                  mean=meanvec,
                                                  std=stdvec)
    val_data = data.DataLoader(val_dataset.transform(transform_val),
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_dataset, val_data, val_batch_fn


def validate(val_loader, net, val_metric, device, flip_test=False):
    val_dataset, val_data, val_batch_fn = val_loader
    val_metric.reset()

    for batch in tqdm(val_data):
        data, scale, center, score, imgid = val_batch_fn(batch, device)

        outputs = [net(X) for X in data]
        if flip_test:
            data_flip = [X.flip(3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip) / 2 for o, o_flip in zip(outputs, outputs_flipback)]

        if len(outputs) > 1:
            outputs_stack = torch.cat([o.to(torch.device('cpu')) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].to(torch.device('cpu'))

        preds, maxvals = get_final_preds(outputs_stack, center.numpy(), scale.numpy())
        val_metric.update(preds, maxvals, score, imgid)

    val_metric.get()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation a model for pose estimation.')
    parser.add_argument('--data-dir', type=str, default='~/.torch/datasets/coco',
                        help='training and validation pictures to use.')
    parser.add_argument('--num-joints', type=int, default=17,
                        help='Number of joints to detect')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--cuda', action='store_true', help='Training with GPUs.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--model', type=str, default='simple_pose_resnet18_v1b',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=str, default='256,192',
                        help='size of the input image size. default is 256,192')
    parser.add_argument('--params-file', type=str,
                        help='local parameters to load.')
    parser.add_argument('--flip-test', action='store_true',
                        help='Whether to flip test input to ensemble results.')
    parser.add_argument('--mean', type=str, default='0.485,0.456,0.406',
                        help='mean vector for normalization')
    parser.add_argument('--std', type=str, default='0.229,0.224,0.225',
                        help='std vector for normalization')
    parser.add_argument('--score-threshold', type=float, default=0,
                        help='threshold value for predicted score.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    device = torch.device('cpu')
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device('cuda:0')

    input_size = [int(i) for i in args.input_size.split(',')]
    val_list = get_dataloader(args.data_dir, args.batch_size, args.num_workers,
                              input_size, args.mean, args.std)
    val_metric = COCOKeyPointsMetric(val_list[0], 'coco_keypoints',
                                     data_shape=tuple(input_size),
                                     in_vis_thresh=args.score_threshold)
    use_pretrained = True if not args.params_file else False
    net = model_zoo.get_model(args.model, num_joints=args.num_joints, pretrained=use_pretrained).to(device)
    net.eval()

    validate(val_list, net, val_metric, device, args.flip_test)
