#!/bin/bash
#SBATCH --job-name=market_gpt_btc      # Job name
#SBATCH --partition=medium              # Partition (medium = 3 days)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPUs per task
#SBATCH --gres=gpu:a100:1               # Request 1 A100 GPU
#SBATCH --mem=64G                       # Memory
#SBATCH --time=3-00:00:00               # Time limit (3 days)
#SBATCH --output=logs/train_%j.out      # Standard output log
#SBATCH --error=logs/train_%j.err       # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=your_email@domain.com  # Your email

# Script de entrenamiento de Market GPT en A100
# Para Bitcoin con velas de 1 minuto

echo "================================================"
echo "Market GPT Training - Bitcoin 1min"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "================================================"

# Load modules (adjust for CESGA environment)
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

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/crypto_cache
mkdir -p results

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Print Python environment
echo "Python environment:"
which python
python --version
echo ""

# Install required packages if needed (first run only)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install numpy pandas scikit-learn requests yfinance tqdm

# Run training script
echo "Starting training..."
python train_bitcoin.py \
    --config configs/bitcoin_gpt_small.json \
    --output_dir checkpoints/bitcoin_run_${SLURM_JOB_ID} \
    --log_dir logs/bitcoin_run_${SLURM_JOB_ID} \
    --device cuda \
    --mixed_precision \
    --gradient_checkpointing

# Check exit status
EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

# Save job info
echo "Job ID: $SLURM_JOB_ID" > results/job_${SLURM_JOB_ID}_info.txt
echo "Node: $SLURM_NODELIST" >> results/job_${SLURM_JOB_ID}_info.txt
echo "Exit code: $EXIT_CODE" >> results/job_${SLURM_JOB_ID}_info.txt
echo "Start: $(date)" >> results/job_${SLURM_JOB_ID}_info.txt

exit $EXIT_CODE
