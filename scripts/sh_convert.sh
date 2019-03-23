#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
# #----------------------------cifar10----------------------------
#python ../utils/gluon2torch.py --name CIFAR_ResNet20_v1
#python ../utils/gluon2torch.py --name CIFAR_ResNet56_v1
#python ../utils/gluon2torch.py --name CIFAR_ResNet110_v1
#python ../utils/gluon2torch.py --name CIFAR_ResNet20_v2
#python ../utils/gluon2torch.py --name CIFAR_ResNet56_v2
#python ../utils/gluon2torch.py --name CIFAR_ResNet110_v2
#python ../utils/gluon2torch.py --name CIFAR_WideResNet16_10
#python ../utils/gluon2torch.py --name CIFAR_WideResNet28_10
#python ../utils/gluon2torch.py --name CIFAR_WideResNet40_8
#python ../utils/gluon2torch.py --name CIFAR_ResNeXt29_16x64d


# #----------------------------imagenet----------------------------
## resnet
#python ../utils/gluon2torch.py --name ResNet18_v1
#python ../utils/gluon2torch.py --name ResNet34_v1
#python ../utils/gluon2torch.py --name ResNet50_v1
#python ../utils/gluon2torch.py --name ResNet101_v1
#python ../utils/gluon2torch.py --name ResNet152_v1
#python ../utils/gluon2torch.py --name ResNet18_v1b
#python ../utils/gluon2torch.py --name ResNet34_v1b
#python ../utils/gluon2torch.py --name ResNet50_v1b
#python ../utils/gluon2torch.py --name ResNet50_v1b_gn
#python ../utils/gluon2torch.py --name ResNet101_v1b
#python ../utils/gluon2torch.py --name ResNet152_v1b
#python ../utils/gluon2torch.py --name ResNet50_v1c
#python ../utils/gluon2torch.py --name ResNet101_v1c
#python ../utils/gluon2torch.py --name ResNet152_v1c
#python ../utils/gluon2torch.py --name ResNet18_v2
#python ../utils/gluon2torch.py --name ResNet34_v2
#python ../utils/gluon2torch.py --name ResNet50_v2
#python ../utils/gluon2torch.py --name ResNet101_v2
#python ../utils/gluon2torch.py --name ResNet152_v2

python ../utils/gluon2torch.py --name ResNet50_v1s

## resnext
#python ../utils/gluon2torch.py --name ResNext50_32x4d
#python ../utils/gluon2torch.py --name ResNext101_32x4d
#python ../utils/gluon2torch.py --name SE_ResNext50_32x4d
#python ../utils/gluon2torch.py --name SE_ResNext101_32x4d
#python ../utils/gluon2torch.py --name SE_ResNext101_64x4d

## mobilenet
#python ../utils/gluon2torch.py --name MobileNet1.0
#python ../utils/gluon2torch.py --name MobileNet0.75
#python ../utils/gluon2torch.py --name MobileNet0.5
#python ../utils/gluon2torch.py --name MobileNet0.25
#python ../utils/gluon2torch.py --name MobileNetV2_1.0
#python ../utils/gluon2torch.py --name MobileNetV2_0.75
#python ../utils/gluon2torch.py --name MobileNetV2_0.5
#python ../utils/gluon2torch.py --name MobileNetV2_0.25

## vgg
#python ../utils/gluon2torch.py --name VGG11
#python ../utils/gluon2torch.py --name VGG13
#python ../utils/gluon2torch.py --name VGG16
#python ../utils/gluon2torch.py --name VGG19
#python ../utils/gluon2torch.py --name VGG11_bn
#python ../utils/gluon2torch.py --name VGG13_bn
#python ../utils/gluon2torch.py --name VGG16_bn
#python ../utils/gluon2torch.py --name VGG19_bn

## squeezenet
#python ../utils/gluon2torch.py --name SqueezeNet1.0
#python ../utils/gluon2torch.py --name SqueezeNet1.1

## densenet
#python ../utils/gluon2torch.py --name DenseNet121
#python ../utils/gluon2torch.py --name DenseNet161
#python ../utils/gluon2torch.py --name DenseNet169
#python ../utils/gluon2torch.py --name DenseNet201

## others
#python ../utils/gluon2torch.py --name AlexNet
#python ../utils/gluon2torch.py --name darknet53
#python ../utils/gluon2torch.py --name InceptionV3
#python ../utils/gluon2torch.py --name SENet_154


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
# #----------------------------voc----------------------------
## ssd
#python ../utils/gluon2torch.py --name ssd_300_vgg16_atrous_voc --base
#python ../utils/gluon2torch.py --name ssd_512_vgg16_atrous_voc --base
#python ../utils/gluon2torch.py --name ssd_512_resnet50_v1_voc --base --reorder
#python ../utils/gluon2torch.py --name ssd_512_mobilenet1.0_voc --base --reorder

## yolo
#python ../utils/gluon2torch.py --name yolo3_darknet53_voc --base
#python ../utils/gluon2torch.py --name yolo3_mobilenet1.0_voc --base


# #----------------------------coco----------------------------
## ssd
#python ../utils/gluon2torch.py --name ssd_300_vgg16_atrous_coco --base
#python ../utils/gluon2torch.py --name ssd_512_vgg16_atrous_coco --base
#python ../utils/gluon2torch.py --name ssd_512_resnet50_v1_coco --base --reorder
#python ../utils/gluon2torch.py --name ssd_512_mobilenet1.0_coco --base --reorder

## yolo
#python ../utils/gluon2torch.py --name yolo3_darknet53_coco --base
#python ../utils/gluon2torch.py --name yolo3_mobilenet1.0_coco --base


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
# #----------------------------ade20k----------------------------
#python ../utils/gluon2torch.py --name fcn_resnet50_ade --base
#python ../utils/gluon2torch.py --name fcn_resnet101_ade --base
#python ../utils/gluon2torch.py --name psp_resnet50_ade --base
#python ../utils/gluon2torch.py --name psp_resnet101_ade --base
#python ../utils/gluon2torch.py --name deeplab_resnet50_ade --base
#python ../utils/gluon2torch.py --name deeplab_resnet101_ade --base


# #----------------------------coco----------------------------
#python ../utils/gluon2torch.py --name fcn_resnet101_coco --base
#python ../utils/gluon2torch.py --name psp_resnet101_coco --base
#python ../utils/gluon2torch.py --name deeplab_resnet101_coco --base


# #----------------------------voc----------------------------
#python ../utils/gluon2torch.py --name fcn_resnet101_voc --base
#python ../utils/gluon2torch.py --name psp_resnet101_voc --base
#python ../utils/gluon2torch.py --name deeplab_resnet101_voc --base
#python ../utils/gluon2torch.py --name deeplab_resnet152_voc --base
#python ../utils/gluon2torch.py --name psp_resnet101_citys --base


# -----------------------------------------------------------------------------
# Pose Estimation
# -----------------------------------------------------------------------------
#python ../utils/gluon2torch.py --name simple_pose_resnet18_v1b
#python ../utils/gluon2torch.py --name simple_pose_resnet50_v1b
#python ../utils/gluon2torch.py --name simple_pose_resnet50_v1d
#python ../utils/gluon2torch.py --name simple_pose_resnet101_v1b
#python ../utils/gluon2torch.py --name simple_pose_resnet101_v1d
#python ../utils/gluon2torch.py --name simple_pose_resnet152_v1b
#python ../utils/gluon2torch.py --name simple_pose_resnet152_v1d
