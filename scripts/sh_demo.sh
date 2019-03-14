#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
# #----------------------------cifar10----------------------------
#python demo/demo_cifar10.py --model CIFAR_ResNet20_v1
#python demo/demo_cifar10.py --model CIFAR_ResNet56_v1
#python demo/demo_cifar10.py --model CIFAR_ResNet110_v1
#python demo/demo_cifar10.py --model CIFAR_ResNet20_v2
#python demo/demo_cifar10.py --model CIFAR_ResNet56_v2
#python demo/demo_cifar10.py --model CIFAR_ResNet110_v2
#python demo/demo_cifar10.py --model CIFAR_WideResNet16_10
#python demo/demo_cifar10.py --model CIFAR_WideResNet28_10
#python demo/demo_cifar10.py --model CIFAR_WideResNet40_8
#python demo/demo_cifar10.py --model CIFAR_ResNeXt29_16x64d


# #----------------------------imagenet----------------------------
## resnet
#python demo/demo_imagenet.py --model ResNet18_v1
#python demo/demo_imagenet.py --model ResNet34_v1
#python demo/demo_imagenet.py --model ResNet50_v1
#python demo/demo_imagenet.py --model ResNet101_v1
#python demo/demo_imagenet.py --model ResNet152_v1
#python demo/demo_imagenet.py --model ResNet18_v1b
#python demo/demo_imagenet.py --model ResNet34_v1b
#python demo/demo_imagenet.py --model ResNet50_v1b
#python demo/demo_imagenet.py --model ResNet50_v1b_gn
#python demo/demo_imagenet.py --model ResNet101_v1b
#python demo/demo_imagenet.py --model ResNet152_v1b
#python demo/demo_imagenet.py --model ResNet50_v1c
#python demo/demo_imagenet.py --model ResNet101_v1c
#python demo/demo_imagenet.py --model ResNet152_v1c
#python demo/demo_imagenet.py --model ResNet18_v2
#python demo/demo_imagenet.py --model ResNet34_v2
#python demo/demo_imagenet.py --model ResNet50_v2
#python demo/demo_imagenet.py --model ResNet101_v2
#python demo/demo_imagenet.py --model ResNet152_v2

## resnext
#python demo/demo_imagenet.py --model ResNext50_32x4d
#python demo/demo_imagenet.py --model ResNext101_32x4d
#python demo/demo_imagenet.py --model SE_ResNext50_32x4d
#python demo/demo_imagenet.py --model SE_ResNext101_32x4d
#python demo/demo_imagenet.py --model SE_ResNext101_64x4d

## mobilenet
#python demo/demo_imagenet.py --model MobileNet1.0
#python demo/demo_imagenet.py --model MobileNet0.75
#python demo/demo_imagenet.py --model MobileNet0.5
#python demo/demo_imagenet.py --model MobileNet0.25
#python demo/demo_imagenet.py --model MobileNetV2_1.0
#python demo/demo_imagenet.py --model MobileNetV2_0.75
#python demo/demo_imagenet.py --model MobileNetV2_0.5
#python demo/demo_imagenet.py --model MobileNetV2_0.25

## vgg
#python demo/demo_imagenet.py --model VGG11
#python demo/demo_imagenet.py --model VGG13
#python demo/demo_imagenet.py --model VGG16
#python demo/demo_imagenet.py --model VGG19
#python demo/demo_imagenet.py --model VGG11_bn
#python demo/demo_imagenet.py --model VGG13_bn
#python demo/demo_imagenet.py --model VGG16_bn
#python demo/demo_imagenet.py --model VGG19_bn

## squeezenet
#python demo/demo_imagenet.py --model SqueezeNet1.0
#python demo/demo_imagenet.py --model SqueezeNet1.1

## densenet
#python demo/demo_imagenet.py --model DenseNet121
#python demo/demo_imagenet.py --model DenseNet161
#python demo/demo_imagenet.py --model DenseNet169
#python demo/demo_imagenet.py --model DenseNet201

## others
#python demo/demo_imagenet.py --model AlexNet
#python demo/demo_imagenet.py --model darknet53
#python demo/demo_imagenet.py --model InceptionV3
#python demo/demo_imagenet.py --model SENet_154


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
# #----------------------------voc----------------------------
## ssd
#python demo/demo_ssd.py --network ssd_300_vgg16_atrous_voc
#python demo/demo_ssd.py --network ssd_512_vgg16_atrous_voc
#python demo/demo_ssd.py --network ssd_512_resnet50_v1_voc
#python demo/demo_ssd.py --network ssd_512_mobilenet1.0_voc

## yolo
#python demo/demo_yolo.py --network yolo3_darknet53_voc
#python demo/demo_yolo.py --network yolo3_mobilenet1.0_voc


# #----------------------------coco----------------------------
## ssd
#python demo/demo_ssd.py --network ssd_300_vgg16_atrous_coco
#python demo/demo_ssd.py --network ssd_512_vgg16_atrous_coco
#python demo/demo_ssd.py --network ssd_512_resnet50_v1_coco
#python demo/demo_ssd.py --network ssd_512_mobilenet1.0_coco

## yolo
#python demo/demo_yolo.py --network yolo3_darknet53_coco
#python demo/demo_yolo.py --network yolo3_mobilenet1.0_coco


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
# #----------------------------ade20k----------------------------
#python demo/demo_segmentation.py --model fcn_resnet50_ade
#python demo/demo_segmentation.py --model fcn_resnet101_ade
#python demo/demo_segmentation.py --model psp_resnet50_ade
#python demo/demo_segmentation.py --model psp_resnet101_ade
#python demo/demo_segmentation.py --model deeplab_resnet50_ade
#python demo/demo_segmentation.py --model deeplab_resnet101_ade


# #----------------------------coco----------------------------
#python demo/demo_segmentation.py --model fcn_resnet101_coco
#python demo/demo_segmentation.py --model psp_resnet101_coco
#python demo/demo_segmentation.py --model deeplab_resnet101_coco


# #----------------------------voc----------------------------
#python demo/demo_segmentation.py --model fcn_resnet101_voc
#python demo/demo_segmentation.py --model psp_resnet101_voc
#python demo/demo_segmentation.py --model deeplab_resnet101_voc
#python demo/demo_segmentation.py --model deeplab_resnet152_voc
#python demo/demo_segmentation.py --model psp_resnet101_citys


# -----------------------------------------------------------------------------
# Pose Estimation
# -----------------------------------------------------------------------------
#python demo/demo_pose.py --pose-model simple_pose_resnet18_v1b
#python demo/demo_pose.py --pose-model simple_pose_resnet50_v1b
#python demo/demo_pose.py --pose-model simple_pose_resnet50_v1d
#python demo/demo_pose.py --pose-model simple_pose_resnet101_v1b
#python demo/demo_pose.py --pose-model simple_pose_resnet101_v1d
#python demo/demo_pose.py --pose-model simple_pose_resnet152_v1b
python demo/demo_pose.py --pose-model simple_pose_resnet152_v1d
