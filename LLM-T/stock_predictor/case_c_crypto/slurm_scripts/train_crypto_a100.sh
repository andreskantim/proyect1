#!/bin/bash
#SBATCH --job-name=case_c               # Job name
#SBATCH --partition=medium              # Partition (3 days)
#SBATCH --nodes=1                       # Nodes
#SBATCH --ntasks=1                      # Tasks
#SBATCH --cpus-per-task=64              # CPUs (required: 32 per GPU × 2 GPUs = 64)
#SBATCH --gres=gpu:a100:2               # 2 A100 GPUs
#SBATCH --mem=128G                      # Memory (64G per GPU)
#SBATCH --time=2-00:00:00               # 2 days limit
#SBATCH --output=../logs/crypto_%j.out  # Output log
#SBATCH --error=../logs/crypto_%j.err   # Error log
# Email notifications disabled (uncomment and set your email if desired)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@domain.com

# Change to case_c_crypto directory
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_c_crypto

echo "================================================"
echo "Case C: Crypto Prototype Training"
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
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p ../checkpoints/case_c_crypto
mkdir -p data/crypto_multi_cache

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run training
echo "Starting training..."

python train_crypto.py \
    --config configs/crypto_prototype.json \
    --output_dir ../checkpoints/case_c_crypto/run_${SLURM_JOB_ID} \
    --device cuda \
    --num-gpus 2 \
    --gradient_checkpointing

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

# Save job info
echo "Job ID: $SLURM_JOB_ID" > ../checkpoints/case_c_crypto/run_${SLURM_JOB_ID}/job_info.txt
echo "Node: $SLURM_NODELIST" >> ../checkpoints/case_c_crypto/run_${SLURM_JOB_ID}/job_info.txt
echo "Exit code: $EXIT_CODE" >> ../checkpoints/case_c_crypto/run_${SLURM_JOB_ID}/job_info.txt

exit $EXIT_CODE
