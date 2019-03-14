#!/usr/bin/env bash

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
python ../utils/gluon2torch.py --name VGG13_bn
#python ../utils/gluon2torch.py --name VGG16_bn
#python ../utils/gluon2torch.py --name VGG19_bn