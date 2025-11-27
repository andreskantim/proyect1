#!/bin/bash
#SBATCH --job-name=test_t4              # Job name
#SBATCH --partition=viz                 # Partition with T4 GPUs
#SBATCH --nodes=1                       # Nodes
#SBATCH --ntasks=1                      # Tasks
#SBATCH --cpus-per-task=8               # CPUs
#SBATCH --gres=gpu:t4:1                 # 1 T4 GPU
#SBATCH --mem=32G                       # Memory
#SBATCH --time=4:00:00                  # 4 hours limit
#SBATCH --output=../logs/test_t4_%j.out # Output log
#SBATCH --error=../logs/test_t4_%j.err  # Error log

# Change to case_c_crypto directory
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_c_crypto

echo "================================================"
echo "Case C: Crypto Test on T4 GPU"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
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

# Save job info
if [ -d "../checkpoints/case_c_crypto_t4_test" ]; then
    echo "Job ID: $SLURM_JOB_ID" > ../checkpoints/case_c_crypto_t4_test/job_info.txt
    echo "Node: $SLURM_NODELIST" >> ../checkpoints/case_c_crypto_t4_test/job_info.txt
    echo "Exit code: $EXIT_CODE" >> ../checkpoints/case_c_crypto_t4_test/job_info.txt
fi

exit $EXIT_CODE
