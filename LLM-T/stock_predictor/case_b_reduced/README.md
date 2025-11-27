# Case B: Reduced Multi-Market Training (100 assets)

## Overview

Train MarketGPT on **100 curated assets** across 4 major markets using **2×A100 GPUs**.

## Assets Coverage

- **US Stocks**: 50 (top tech, finance, healthcare)
- **Crypto**: 20 (major cryptocurrencies)
- **Commodities**: 15 (gold, oil, agriculture)
- **Emerging Markets**: 15 (major EM ETFs/companies)

**Total**: 100 assets across 4 categories

## Hardware Requirements

- **GPUs**: 2×A100-40GB
- **CPUs**: 64
- **RAM**: 128GB
- **Storage**: ~10GB for data cache
- **Time**: 3-5 days

## Model Configuration

- **Parameters**: ~45M
- `d_model=512`, `num_layers=8`, `num_heads=8`
- `context_length=256`, `vocab_size=2048`
- `batch_size=32` (effective: 64 with 2 GPUs)

## Quick Start

### Individual Launch

```bash
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh
```

### Parallel Launch (with Case A)

```bash
cd ..  # Go to stock_predictor/
./launch_parallel_training.sh
```

## Monitor Training

```bash
# Check job status
squeue -u $(whoami)

# View logs
tail -f logs/reduced_*.out

# Monitor training
../monitor_training.sh  # If launched in parallel
```

## Outputs

Checkpoints saved to: `checkpoints/reduced_YYYYMMDD_HHMMSS/`

- `best_model.pt` - Best model by validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `tokenizer.pkl` - Fitted tokenizer
- `asset_info.json` - Asset ID mappings
- `training_log.json` - Full training history

## Files

```
case_b_reduced/
├── multi_market_loader.py      # Data loader for 100 assets
├── train_reduced.py            # Training script (multi-GPU)
├── configs/
│   └── reduced_config.json     # Model & training config
├── slurm_scripts/
│   └── train_reduced_a100.sh   # SLURM submission script
├── checkpoints/                # Training outputs
└── logs/                       # SLURM logs
```

## Expected Performance

- **Training time**: 3-5 days
- **Val loss**: 2.0-3.0 (after convergence)
- **Val accuracy**: 45-55%
- **GPU utilization**: ~95%
- **Memory per GPU**: ~12GB

## Advantages over Case A

- **Faster training**: ~50% less time
- **Lower memory**: Smaller model fits easily
- **Curated assets**: High-quality, liquid assets only
- **Better for prototyping**: Quick iterations

## Next Steps

After training:

1. **Compare with Case A**: Does more data help?
2. **Baseline for experiments**: Use as reference
3. **Fine-tune**: Adapt to specific use cases
4. **Deploy**: Smaller model = faster inference

## Troubleshooting

### OOM Error

Reduce `batch_size` in config:
```json
"training": {
  "batch_size": 24  // or 16
}
```

### Missing Assets

Some assets may be delisted or have sparse data. Check logs:
```bash
grep -i "failed" logs/reduced_*.out
```

### Training Too Fast?

Consider increasing model size:
```json
"model": {
  "d_model": 768,
  "num_layers": 10,
  "num_heads": 12
}
```

## See Also

- **Main guide**: `../MULTI_GPU_TRAINING_GUIDE.md`
- **Case A**: `../case_a_full_market/README.md` (600 assets)
- **Case C**: `../case_c_crypto/README.md` (crypto only)
