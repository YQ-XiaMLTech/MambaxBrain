#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main.py \
  --modes '{"fingerprint":"test","cognitive":"test","rt":"train"}' \
  --epoch 60 \
  --batch_size 128 \
  --lr 0.0008 \
  --window_size 1 \
  --d_state 150 \
  --d_conv 16 \
  --expand 16 \
  --train_ratio 0.7 \
  --seed 888 \
  --condition "natural"
