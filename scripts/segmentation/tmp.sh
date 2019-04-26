#!/usr/bin/env bash

#export NGPUS=8
#srun --partition=HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
#     --model psp --backbone resnet101 --dataset citys --batch-size 2 --test-batch-size 2 --lr -1 \
#     --base-size 1024 --crop-size 768 --jpu \
#     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20 --aux


export NGPUS=8
srun --partition=HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
     --model bisenet --backbone resnet18 --dataset citys --batch-size 8 --test-batch-size 2 --lr -1 \
     --base-size 1024 --crop-size 768 --ohem \
     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20

#python train_segmentation_pil.py \
#     --model bisenet --backbone resnet18 --dataset citys --batch-size 2 --test-batch-size 2 --lr -1 \
#     --base-size 1024 --crop-size 768 --ohem \
#     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20


#export NGPUS=8
#srun --partition=HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
#     --model bisenet --backbone resnet18 --dataset pascal_paper --batch-size 16 --test-batch-size 2 --lr -1 \
#     --base-size 540 --crop-size 480 \
#     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10


#python train_segmentation_pil.py \
#     --model bisenet --backbone resnet18 --dataset pascal_paper --batch-size 2 --test-batch-size 2 --lr -1 \
#     --base-size 540 --crop-size 480 \
#     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10