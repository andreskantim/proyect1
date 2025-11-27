#!/bin/bash
#SBATCH --job-name=diag_t4             # Job name
#SBATCH --partition=viz                # Partition with T4 GPUs
#SBATCH --nodes=1                      # Nodes
#SBATCH --ntasks=1                     # Tasks
#SBATCH --cpus-per-task=16             # CPUs
#SBATCH --gres=gpu:t4:1                # 1 T4 GPU
#SBATCH --mem=48G                      # Memory
#SBATCH --time=6:00:00                 # 6 hours limit
#SBATCH --output=../logs/diagnostic_t4_%j.out
#SBATCH --error=../logs/diagnostic_t4_%j.err

cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_c_crypto

echo "================================================"
echo "Case C: DIAGNOSTIC TEST on T4 GPU"
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
mkdir -p ../checkpoints/case_c_crypto_diagnostic

# GPU info
echo ""
nvidia-smi
echo ""

# Run training with diagnostic config
echo "Starting diagnostic training (10 epochs, better config)..."

python train_crypto.py \
    --config configs/crypto_t4_diagnostic.json \
    --output_dir ../checkpoints/case_c_crypto_diagnostic \
    --device cuda \
    --num-gpus 1

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Diagnostic training finished: exit code $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
