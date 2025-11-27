# Case A: Full Market Training (600 assets)

## Overview

Train MarketGPT on **600 assets** across all major global markets using **2×A100 GPUs**.

## Assets Coverage

- **US Stocks**: 300 (S&P 500 top companies)
- **European Stocks**: 150 (major EU indices)
- **Emerging Markets**: 50 (ETFs + major companies)
- **Commodities**: 30 (metals, energy, agriculture)
- **Crypto**: 70 (top cryptocurrencies)

**Total**: 600 assets across 5 categories

## Hardware Requirements

- **GPUs**: 2×A100-40GB
- **CPUs**: 64
- **RAM**: 128GB
- **Storage**: ~50GB for data cache
- **Time**: 7-10 days

## Model Configuration

- **Parameters**: ~85M
- `d_model=768`, `num_layers=12`, `num_heads=12`
- `context_length=512`, `vocab_size=4096`
- `batch_size=32` (effective: 64 with 2 GPUs)

## Quick Start

### Individual Launch

```bash
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

### Parallel Launch (with Case B)

```bash
cd ..  # Go to stock_predictor/
./launch_parallel_training.sh
```

## Monitor Training

```bash
# Check job status
squeue -u $(whoami)

# View logs
tail -f logs/full_*.out

# Monitor training
../monitor_training.sh  # If launched in parallel
```

## Outputs

Checkpoints saved to: `checkpoints/full_market_YYYYMMDD_HHMMSS/`

- `best_model.pt` - Best model by validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `tokenizer.pkl` - Fitted tokenizer
- `asset_info.json` - Asset ID mappings
- `training_log.json` - Full training history

## Files

```
case_a_full_market/
├── universal_loader.py         # Data loader for 600 assets
├── train_full.py               # Training script (multi-GPU)
├── configs/
│   └── full_market_config.json # Model & training config
├── slurm_scripts/
│   └── train_full_a100.sh      # SLURM submission script
├── checkpoints/                # Training outputs
└── logs/                       # SLURM logs
```

## Expected Performance

- **Training time**: 7-10 days
- **Val loss**: 2.5-3.5 (after convergence)
- **Val accuracy**: 40-50%
- **GPU utilization**: ~95%
- **Memory per GPU**: ~20GB

## Next Steps

After training:

1. **Evaluate**: Compare with Case B and Case C
2. **Predict**: Use for price forecasting
3. **Export**: Convert to ONNX for deployment
4. **Fine-tune**: Retrain on specific markets/assets

## Troubleshooting

### OOM Error

Reduce `batch_size` in config:
```json
"training": {
  "batch_size": 24  // or 16
}
```

### Slow Data Loading

Pre-download data:
```bash
python -c "from universal_loader import UniversalMarketDataLoader; loader = UniversalMarketDataLoader(); loader.load_all_data()"
```

### Training Stalled

Check GPU utilization:
```bash
ssh <node_name>
nvidia-smi -l 1
```

## See Also

- **Main guide**: `../MULTI_GPU_TRAINING_GUIDE.md`
- **Case B**: `../case_b_reduced/README.md` (100 assets)
- **Case C**: `../case_c_crypto/README.md` (crypto only)
