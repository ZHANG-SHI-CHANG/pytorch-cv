#!/usr/bin/env bash

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
python demo/demo_imagenet.py --model VGG13_bn
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