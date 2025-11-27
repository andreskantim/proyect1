# Case C: Crypto Prototype (20 assets)

## Overview

Train MarketGPT on **20 major cryptocurrencies** using **2×A100 GPUs**. This is the fastest case, ideal for prototyping and testing.

## Assets Coverage

**20 Major Cryptocurrencies**:
- BTC, ETH, BNB, XRP, ADA, DOGE, SOL, TRX
- DOT, MATIC, LTC, SHIB, AVAX, UNI, LINK
- ATOM, XLM, BCH, FIL, APT

## Hardware Requirements

- **GPUs**: 2×A100-40GB (single node)
- **CPUs**: 64
- **RAM**: 128GB
- **Storage**: ~5GB for data cache
- **Time**: 1-2 days

## Model Configuration

- **Parameters**: ~25M
- `d_model=256`, `num_layers=6`, `num_heads=8`
- `context_length=128`, `vocab_size=1024`
- `batch_size=32` (effective: 64 with 2 GPUs)

## Quick Start

### Launch Training

```bash
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh
```

### Monitor

```bash
# Check job status
squeue -u $(whoami)

# View logs in real-time
tail -f logs/crypto_*.out

# Check for errors
tail -f logs/crypto_*.err
```

## Files

```
case_c_crypto/
├── crypto_data_loader.py       # Crypto data downloader
├── train_crypto.py             # Training script (multi-GPU)
├── configs/
│   └── crypto_prototype.json   # Model & training config
├── slurm_scripts/
│   └── train_crypto_a100.sh    # SLURM submission script
├── checkpoints/                # Training outputs
└── logs/                       # SLURM logs
```

## Outputs

Checkpoints saved to: `../checkpoints/case_c_crypto/run_<JOB_ID>/`

- `best_model.pt` - Best model by validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `tokenizer.pkl` - Fitted tokenizer
- `data_info.json` - Dataset info
- `training_summary.json` - Final summary

## Expected Performance

- **Training time**: 1-2 days on 2×A100
- **Val loss**: 1.5-2.5 (after convergence)
- **Val accuracy**: 50-60%
- **GPU utilization**: ~95%
- **Memory per GPU**: ~8GB

## Why Case C?

### Advantages:
- **Fast iteration**: 1-2 days vs 7-10 for Case A
- **Cheap**: Lower compute cost
- **Crypto-focused**: Specialized for volatile markets
- **Good baseline**: Test ideas before full training

### Use Cases:
- **Prototyping**: Test new architectures
- **Hyperparameter tuning**: Quick experiments
- **Crypto trading**: Deploy for crypto-only predictions
- **Learning**: Understand the training pipeline

## Recent Fixes

### Fixed Issues:
1. **DatetimeArray.sort() error**: Changed to `sorted()` ✅
2. **Single GPU**: Now uses 2 GPUs with DataParallel ✅
3. **CUDA module error**: Removed incompatible module load ✅

### Current Status:
- ✅ Bug-free
- ✅ Multi-GPU ready
- ✅ Tested on cluster

## Troubleshooting

### OOM Error

Reduce `batch_size` in config:
```json
"training": {
  "batch_size": 16
}
```

### Missing Crypto Data

Some cryptos may have limited history. Check logs:
```bash
grep "days" logs/crypto_*.out
```

### Job Pending

Check GPU availability:
```bash
sinfo -p medium -o "%P %D %N %G"
```

## Next Steps

After training:

1. **Compare with Cases A & B**: Does crypto-specific model perform better on crypto?
2. **Deploy**: Smaller model = faster inference
3. **Fine-tune**: Retrain on specific crypto subset
4. **Extend**: Add more cryptos or technical indicators

## See Also

- **Main guide**: `../TRAINING_GUIDE.md`
- **Case A**: `../case_a_full_market/README.md` (600 assets)
- **Case B**: `../case_b_reduced/README.md` (100 assets)
