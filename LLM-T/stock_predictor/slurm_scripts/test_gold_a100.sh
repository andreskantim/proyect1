#!/bin/bash
#SBATCH --job-name=market_gpt_gold     # Job name
#SBATCH --partition=medium              # Partition (medium = 3 days)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPUs per task
#SBATCH --gres=gpu:a100:1               # Request 1 A100 GPU
#SBATCH --mem=64G                       # Memory
#SBATCH --time=2-00:00:00               # Time limit (2 days)
#SBATCH --output=logs/test_gold_%j.out  # Standard output log
#SBATCH --error=logs/test_gold_%j.err   # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=your_email@domain.com  # Your email

# Script para testing/walk-forward en datos de oro
# Usa modelo pre-entrenado en Bitcoin

echo "================================================"
echo "Market GPT Walk-Forward Testing - Gold 20 years"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "================================================"

# Load modules
module purge
module load cuda/11.8
module load python/3.10

# Activate conda environment
source ~/.bashrc
conda activate llm-training

# Set environment variables
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/gold_cache
mkdir -p results/gold_test

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run walk-forward testing on gold
echo "Starting walk-forward testing on gold..."
python test_gold_walkforward.py \
    --checkpoint checkpoints/bitcoin_best/best_initial.pt \
    --config configs/bitcoin_gpt_small.json \
    --output_dir results/gold_test_${SLURM_JOB_ID} \
    --window_size 10080 \
    --fine_tune_epochs 3 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Testing finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

# Save results summary
echo "Job ID: $SLURM_JOB_ID" > results/gold_test_${SLURM_JOB_ID}/job_info.txt
echo "Node: $SLURM_NODELIST" >> results/gold_test_${SLURM_JOB_ID}/job_info.txt
echo "Exit code: $EXIT_CODE" >> results/gold_test_${SLURM_JOB_ID}/job_info.txt

exit $EXIT_CODE
