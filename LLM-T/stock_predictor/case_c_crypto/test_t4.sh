#!/bin/bash

echo "================================================"
echo "Case C: Crypto Test on T4 GPU"
echo "================================================"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================"

# Activate environment
source ~/.bashrc
conda activate llm-training

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8

# Create directories
mkdir -p logs
mkdir -p ../checkpoints/case_c_crypto_t4_test

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run training
echo "Starting test training on T4..."

python train_crypto.py \
    --config configs/crypto_t4_test.json \
    --output_dir ../checkpoints/case_c_crypto_t4_test \
    --device cuda \
    --num-gpus 1

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
