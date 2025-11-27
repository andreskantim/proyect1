#!/bin/bash
#SBATCH --job-name=case_b               # Job name
#SBATCH --partition=long                # Partition (7 days max)
#SBATCH --nodes=1                       # Nodes
#SBATCH --ntasks=1                      # Tasks
#SBATCH --cpus-per-task=64              # CPUs (required: 32 per GPU × 2 GPUs = 64)
#SBATCH --gres=gpu:a100:2               # 2 A100 GPUs
#SBATCH --mem=128G                      # Memory (64G per GPU)
#SBATCH --time=5-00:00:00               # 5 days limit
#SBATCH --output=logs/reduced_%j.out    # Output log
#SBATCH --error=logs/reduced_%j.err     # Error log
# Email notifications disabled (uncomment and set your email if desired)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@domain.com

echo "================================================"
echo "Case B: Reduced Multi-Market Training (100 assets)"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "================================================"

# Load modules
module purge
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate llm-training

# Environment variables for multi-GPU
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p ../checkpoints/case_b_reduced
mkdir -p ../data/reduced_cache

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "PyTorch CUDA info:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs detected: {torch.cuda.device_count()}')"
echo ""

# Run training
echo "Starting training on 2x A100 GPUs..."
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_b_reduced

python train_reduced.py \
    --config configs/reduced_config.json \
    --output_dir ../checkpoints/case_b_reduced/run_${SLURM_JOB_ID} \
    --device cuda \
    --num-gpus 2

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

# Save job info
echo "Job ID: $SLURM_JOB_ID" > ../checkpoints/case_b_reduced/run_${SLURM_JOB_ID}/job_info.txt
echo "Node: $SLURM_NODELIST" >> ../checkpoints/case_b_reduced/run_${SLURM_JOB_ID}/job_info.txt
echo "GPUs: 2x A100" >> ../checkpoints/case_b_reduced/run_${SLURM_JOB_ID}/job_info.txt
echo "Exit code: $EXIT_CODE" >> ../checkpoints/case_b_reduced/run_${SLURM_JOB_ID}/job_info.txt

exit $EXIT_CODE
