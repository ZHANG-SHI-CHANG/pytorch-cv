#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
# #----------------------------cifar10----------------------------
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet20_v1 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet56_v1 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet110_v1 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet20_v2 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet56_v2 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet110_v2 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet20_v2 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_WideResNet16_10 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_WideResNet28_10 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_WideResNet40_8 --batch-size 8 --cuda

#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNeXt29_16x64d --batch-size 8 --cuda



# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
# #----------------------------voc----------------------------
## ssd
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset voc --data-shape 300 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset voc --data-shape 512 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network resnet50_v1 --batch-size 4 --dataset voc --data-shape 512 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 512 --cuda


## yolo
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset voc --data-shape 320 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset voc --data-shape 416 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 320 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset voc --data-shape 416 --cuda


# #----------------------------coco----------------------------
## ssd
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset coco --data-shape 300 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network vgg16_atrous --batch-size 4 --dataset coco --data-shape 512 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network resnet50_v1 --batch-size 4 --dataset coco --data-shape 512 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_ssd.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 512 --cuda


## yolo
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 320 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 416 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network darknet53 --batch-size 4 --dataset coco --data-shape 608 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 320 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 416 --cuda
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_yolo.py --network mobilenet1.0 --batch-size 4 --dataset coco --data-shape 608 --cuda
