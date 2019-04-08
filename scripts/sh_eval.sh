#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
# #----------------------------cifar10----------------------------
#python eval/eval_cifar10.py --network CIFAR_ResNet20_v1 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet56_v1 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet110_v1 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet20_v2 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet56_v2 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet110_v2 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNet20_v2 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_WideResNet16_10 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_WideResNet28_10 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_WideResNet40_8 --batch-size 8 --cuda
#python eval/eval_cifar10.py --network CIFAR_ResNeXt29_16x64d --batch-size 8 --cuda


# #----------------------------imagenet----------------------------
## resnet
#python eval/eval_imagenet.py --model ResNet18_v1 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet34_v1 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet50_v1 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet101_v1 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet152_v1 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet18_v1b --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet34_v1b --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet50_v1b --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet50_v1b_gn --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet101_v1b --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet152_v1b --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet50_v1c --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet101_v1c --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet152_v1c --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet18_v2 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet34_v2 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet50_v2 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet101_v2 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNet152_v2 --batch-size 64 --cuda

## resnext
#python eval/eval_imagenet.py --model ResNext50_32x4d --batch-size 64 --cuda
#python eval/eval_imagenet.py --model ResNext101_32x4d --batch-size 64 --cuda
#python eval/eval_imagenet.py --model SE_ResNext50_32x4d --batch-size 64 --cuda
#python eval/eval_imagenet.py --model SE_ResNext101_32x4d --batch-size 64 --cuda
#python eval/eval_imagenet.py --model SE_ResNext101_64x4d --batch-size 64 --cuda

## mobilenet
#python eval/eval_imagenet.py --model MobileNet1.0 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNet0.75 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNet0.5 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNet0.25 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNetV2_1.0 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNetV2_0.75 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNetV2_0.5 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model MobileNetV2_0.25 --batch-size 64 --cuda

## vgg
#python eval/eval_imagenet.py --model VGG11 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG13 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG16 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG19 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG11_bn --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG13_bn --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG16_bn --batch-size 64 --cuda
#python eval/eval_imagenet.py --model VGG19_bn --batch-size 64 --cuda

## squeezenet
#python eval/eval_imagenet.py --model SqueezeNet1.0 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model SqueezeNet1.1 --batch-size 64 --cuda

## densenet
#python eval/eval_imagenet.py --model DenseNet121 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model DenseNet161 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model DenseNet169 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model DenseNet201 --batch-size 64 --cuda

## others
#python eval/eval_imagenet.py --model AlexNet --batch-size 64 --cuda
#python eval/eval_imagenet.py --model darknet53 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model InceptionV3 --batch-size 64 --cuda
#python eval/eval_imagenet.py --model SENet_154 --batch-size 64 --cuda


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
# #----------------------------voc----------------------------
## ssd
#python eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset voc --data-shape 300 --cuda
#python eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset voc --data-shape 512 --cuda
#python eval/eval_ssd.py --network resnet50_v1 --batch-size 4 --dataset voc --data-shape 512 --cuda
#python eval/eval_ssd.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 512 --cuda


## yolo
#python eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset voc --data-shape 320 --cuda
#python eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset voc --data-shape 416 --cuda
#python eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 320 --cuda
#python eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 416 --cuda


# #----------------------------coco----------------------------
## ssd
#python eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset coco --data-shape 300 --cuda
#python eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset coco --data-shape 512 --cuda
#python eval/eval_ssd.py --network resnet50_v1 --batch-size 4 --dataset coco --data-shape 512 --cuda
#python eval/eval_ssd.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 512 --cuda


## yolo
#python eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 320 --cuda
#python eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 416 --cuda
#python eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 608 --cuda
#python eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 320 --cuda
#python eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 416 --cuda
#python eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 608 --cuda


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
# #----------------------------ade20k----------------------------
#python eval/eval_segmentation.py --model_name fcn_resnet50_ade --dataset ade20k --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name fcn_resnet101_ade --dataset ade20k --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name psp_resnet50_ade --dataset ade20k --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name psp_resnet101_ade --dataset ade20k --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name deeplab_resnet50_ade --dataset ade20k --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name deeplab_resnet101_ade --dataset ade20k --batch-size 1 -j 4 --cuda


# #----------------------------coco----------------------------
#python eval/eval_segmentation.py --model_name fcn_resnet101_coco --dataset coco --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name psp_resnet101_coco --dataset coco --batch-size 1 -j 4 --cuda
#python eval/eval_segmentation.py --model_name deeplab_resnet101_coco --dataset coco --batch-size 1 -j 4 --cuda



# -----------------------------------------------------------------------------
# Pose Estimation
# -----------------------------------------------------------------------------
## simple pose
#python eval/eval_pose.py --model simple_pose_resnet18_v1b --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet18_v1b --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet18_v1b --batch-size 4 --input-size 128,96 --cuda
#python eval/eval_pose.py --model simple_pose_resnet18_v1b --batch-size 4 --input-size 128,96 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet50_v1b --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet50_v1b --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet50_v1d --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet50_v1d --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet101_v1b --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet101_v1b --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet101_v1d --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet101_v1d --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet152_v1b --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet152_v1b --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet152_v1d --batch-size 4 --cuda
#python eval/eval_pose.py --model simple_pose_resnet152_v1d --batch-size 4 --cuda --flip-test
#python eval/eval_pose.py --model simple_pose_resnet152_v1d --batch-size 4 --input-size 384,288 --cuda
#python eval/eval_pose.py --model simple_pose_resnet152_v1d --batch-size 4 --input-size 384,288 --cuda --flip-test