#!/bin/bash
#SBATCH --job-name=multitask_t4          # Job name
#SBATCH --partition=viz                  # Partition with T4 GPUs
#SBATCH --nodes=1                        # Nodes
#SBATCH --ntasks=1                       # Tasks
#SBATCH --cpus-per-task=16               # CPUs
#SBATCH --gres=gpu:t4:1                  # 1 T4 GPU
#SBATCH --mem=48G                        # Memory
#SBATCH --time=2:00:00                   # 2 hours limit
#SBATCH --output=../logs/multitask_t4_%j.out
#SBATCH --error=../logs/multitask_t4_%j.err

cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_c_crypto

echo "================================================"
echo "Case C: MULTI-TASK TEST on T4 GPU"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

# Activate environment
source ~/.bashrc
conda activate llm-training

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p ../checkpoints/case_c_multitask_daily_test

# GPU info
echo ""
nvidia-smi
echo ""

# Run training with multi-task config
echo "Starting multi-task training test (daily data)..."

python train_multitask.py \
    --config configs/crypto_multitask_daily.json \
    --output_dir ../checkpoints/case_c_multitask_daily_test \
    --device cuda \
    --num-gpus 1

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Multi-task training test finished: exit code $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
