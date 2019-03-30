#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
# #----------------------------cifar10----------------------------
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train/train_cifar10.py --model CIFAR_ResNet20_v1 \
#     --num-epochs 200 -j 4 --batch-size 128 --wd 0.0001 --lr 0.4 --lr-decay 0.1 --lr-decay-epoch 100,150 --cuda



# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
#export NGPUS=8
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train/train_segmentation.py --dataset ade20k \
#     --model fcn -j 8 --aux --lr 0.08 --checkname res50 --epochs 120 --batch-size 4 --test-batch-size 4 --val-inter 120

