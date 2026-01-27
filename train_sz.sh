#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main_sz.py \
  --epoch 60 \
  --batch_size 128 \
  --lr 0.0008 \
  --train_ratio 0.6 \
  --seed 888 \
