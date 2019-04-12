#!/usr/bin/env bash

#export NGPUS=8
#srun --partition=HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 python -m torch.distributed.launch --nproc_per_node=$NGPUS train/train_segmentation.py \
#     --dataset ade20k --model fcn --backbone resnet50 --aux --lr 0.02 --checkname res50  \
#     --epochs 120 --batch-size 4

#export NGPUS=4
#srun --partition=HA_3D --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=8 python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_segmentation.py --model_name fcn_resnet50_ade --dataset ade20k --batch-size 1 -j 4 --cuda


#python eval/eval_segmentation.py --model_name fcn_resnet50_ade --dataset ade20k --batch-size 1 -j 4 --cuda


#python train/train_segmentation.py \
#     --dataset ade20k --model fcn --backbone resnet50 --aux --lr 0.08 --checkname res50  --epochs 120 --batch-size 1