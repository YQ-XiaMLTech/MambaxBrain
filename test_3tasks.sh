#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MODES='{"fingerprint":"test","cognitive":"test","rt":"test"}'

python test_model.py \
  --modes "$MODES" \
  --run_dir "result_model/3task" \
  --window_size 1 \
  --sequence_length 100 \
  --batch_size 64 \
  --train_ratio 0.7 \
  --seed 123 \
  --condition "natural" \
  --gpu
