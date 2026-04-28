#!/usr/bin/env bash
# Reproduce the three core experiments end-to-end.
#
# Stages:
#   1. Train ResNet-18 on CIFAR-100  (~30 min on a single GPU)
#   2. Cache features from the trained checkpoint
#   3. Run Experiments 1, 2, 3
#
# All outputs land under ./results/

set -e

CKPT_DIR=${CKPT_DIR:-checkpoints/baseline}
EPOCHS=${EPOCHS:-30}
DEVICE=${DEVICE:-cuda}

mkdir -p cache results

echo "==> [1/3] Training ResNet-18 on CIFAR-100 ($EPOCHS epochs)"
python train.py --epochs $EPOCHS --save-dir $CKPT_DIR --device $DEVICE

echo "==> [2/3] Extracting features (test split)"
python extract_features.py \
    --checkpoint $CKPT_DIR/best.pt \
    --split test \
    --out cache/test_features.pt \
    --device $DEVICE

echo "==> [2.1/3] Extracting features (train split)"
python extract_features.py \
    --checkpoint $CKPT_DIR/best.pt \
    --split train \
    --out cache/train_features.pt \
    --device $DEVICE

echo "==> [3/3] Running experiments"
python exp1_null_space.py     --features cache/test_features.pt --out results/exp1
python exp2_bound_comparison.py --features cache/test_features.pt --out results/exp2
python exp3_T_star.py         --features cache/test_features.pt --out results/exp3

echo "==> All done. See results/ for figures and JSON."
