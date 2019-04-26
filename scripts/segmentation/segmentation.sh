#!/usr/bin/env bash

# ---------------------------------train------------------------------
# ---------voc2012----------
# FCN
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
     --model fcn --backbone resnet101 --dataset pascal_paper --batch-size 8 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --jpu \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux


# PSP
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
     --model psp --backbone resnet101 --dataset pascal_paper --batch-size 4 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --dilated \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux

# DeepLabv3



# ---------cityscapes----------
# FCN
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py \
     --model fcn --backbone resnet101 --dataset pascal_paper --batch-size 8 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --jpu \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux