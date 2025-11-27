#!/bin/bash
#SBATCH --job-name=case_a               # Job name
#SBATCH --partition=long                # Partition (7 days max)
#SBATCH --nodes=1                       # Nodes
#SBATCH --ntasks=1                      # Tasks
#SBATCH --cpus-per-task=64              # CPUs (required: 32 per GPU × 2 GPUs = 64)
#SBATCH --gres=gpu:a100:2               # 2 A100 GPUs
#SBATCH --mem=128G                      # Memory (64G per GPU)
#SBATCH --time=7-00:00:00               # 7 days limit (full partition time)
#SBATCH --output=logs/full_%j.out       # Output log
#SBATCH --error=logs/full_%j.err        # Error log

# =============================================================================
# Case A: Full Market Training (600 assets)
# Expected runtime: 7-10 days on 2×A100
# =============================================================================

echo "======================================================================"
echo "Case A: Full Market Training - 600 Assets"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================================================"

# Environment setup
echo ""
echo "🔧 Setting up environment..."
source ~/.bashrc

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate llm-training
    echo "✅ Conda environment activated: llm-training"
else
    echo "⚠️  Conda not found, proceeding without activation"
fi

# Verify GPU availability
echo ""
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Project paths
PROJECT_ROOT="/mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor"
CASE_DIR="${PROJECT_ROOT}/case_a_full_market"
CONFIG="${CASE_DIR}/configs/full_market_config.json"
OUTPUT="${CASE_DIR}/checkpoints/full_market_$(date +%Y%m%d_%H%M%S)"

echo "📁 Project paths:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Case directory: ${CASE_DIR}"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT}"
echo ""

# Change to case directory
cd "${CASE_DIR}" || exit 1

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip list | grep -E "(torch|yfinance|pandas)" || pip install -q -r ../requirements_gpu.txt
echo ""

# Start training
echo "======================================================================"
echo "🚀 Starting Case A training..."
echo "======================================================================"
echo ""

python train_full.py \
    --config "${CONFIG}" \
    --output "${OUTPUT}" \
    --device cuda \
    --num-gpus 2

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved to: ${OUTPUT}"

    # Generate summary
    echo ""
    echo "📊 Training Summary:"
    if [ -f "${OUTPUT}/training_log.json" ]; then
        python -c "
import json
with open('${OUTPUT}/training_log.json') as f:
    log = json.load(f)
    epochs = log['epochs']
    best_epoch = min(epochs, key=lambda x: x['val_loss'])
    print(f'  Total epochs: {len(epochs)}')
    print(f'  Best epoch: {best_epoch[\"epoch\"]}')
    print(f'  Best val loss: {best_epoch[\"val_loss\"]:.4f}')
    print(f'  Best val accuracy: {best_epoch[\"val_accuracy\"]:.2f}%')
"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "📋 Check logs for details:"
    echo "  Output: logs/full_${SLURM_JOB_ID}.out"
    echo "  Error: logs/full_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
