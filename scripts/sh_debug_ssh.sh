#!/usr/bin/env bash


export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS debug/tmp.py
