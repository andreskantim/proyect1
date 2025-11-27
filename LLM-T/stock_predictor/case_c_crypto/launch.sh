#!/bin/bash
# Launch script for Case C: Crypto Prototype

echo "=================================="
echo "Case C: Crypto Prototype Launcher"
echo "=================================="
echo ""

# Check environment
echo "1. Checking environment..."
conda activate llm-training 2>&1 | grep -q "Could not find" && {
    echo "ERROR: llm-training environment not found"
    exit 1
}
echo "   ✓ Environment llm-training found"

# Check Python and PyTorch
echo "2. Checking Python packages..."
python -c "import torch; print(f'   ✓ PyTorch {torch.__version__}')" || {
    echo "ERROR: PyTorch not installed"
    exit 1
}

python -c "import pandas, sklearn, yfinance, tqdm; print('   ✓ All dependencies OK')" || {
    echo "ERROR: Missing dependencies"
    exit 1
}

# Check CUDA availability (if on compute node)
echo "3. Checking CUDA..."
python -c "import torch; print(f'   ✓ CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
echo "4. Creating directories..."
mkdir -p ../data/crypto_multi_cache
mkdir -p ../checkpoints/case_c_crypto
mkdir -p logs
echo "   ✓ Directories created"

# Check config file
echo "5. Checking configuration..."
if [ ! -f "configs/crypto_prototype.json" ]; then
    echo "ERROR: Config file not found"
    exit 1
fi
echo "   ✓ Config file found"

# Display config
echo ""
echo "Configuration:"
python -c "
import json
with open('configs/crypto_prototype.json') as f:
    config = json.load(f)
    print(f\"  Experiment: {config['experiment_name']}\")
    print(f\"  Assets: 20 cryptocurrencies\")
    print(f\"  Model: {config['model']['n_layers']} layers, {config['model']['d_model']} dim\")
    print(f\"  Epochs: {config['training']['epochs']}\")
    print(f\"  Batch size: {config['training']['batch_size']}\")
"

echo ""
echo "=================================="
echo "Ready to submit!"
echo "=================================="
echo ""
echo "Options:"
echo "  1. Submit to SLURM queue"
echo "  2. Test locally (CPU, 1 epoch)"
echo "  3. Cancel"
echo ""
read -p "Select option (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Submitting to SLURM..."
        JOB_ID=$(sbatch slurm_scripts/train_crypto_a100.sh | awk '{print $4}')
        echo ""
        echo "✓ Job submitted with ID: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -j $JOB_ID"
        echo "  tail -f logs/crypto_${JOB_ID}.out"
        echo ""
        echo "Or use:"
        echo "  bash ../slurm_scripts/monitor_job.sh $JOB_ID"
        ;;

    2)
        echo ""
        echo "Running local test (CPU, 1 epoch)..."
        echo ""

        # Create test config
        python -c "
import json
with open('configs/crypto_prototype.json') as f:
    config = json.load(f)
config['training']['epochs'] = 1
config['training']['batch_size'] = 4
config['data']['start_date'] = '2023-01-01'
with open('/tmp/test_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"

        python train_crypto.py \
            --config /tmp/test_config.json \
            --output_dir ../checkpoints/test_local \
            --device cpu

        echo ""
        echo "Test complete! Check ../checkpoints/test_local/"
        ;;

    3)
        echo "Cancelled."
        exit 0
        ;;

    *)
        echo "Invalid option"
        exit 1
        ;;
esac
