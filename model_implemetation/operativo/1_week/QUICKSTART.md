# Quick Start Guide - Bitcoin 1-Week Prediction Model

## Prerequisites

- Python 3.7 or higher
- Access to SLURM cluster with 64+ cores
- Bitcoin hourly candlestick data (3 years)

## Installation

### 1. Install Python dependencies

```bash
cd 1_week
pip install -r requirements.txt
```

### 2. Prepare your data

Your Bitcoin data CSV should have these columns:
- `timestamp` (or `date`)
- `open`
- `high`
- `low`
- `close`
- `volume`

**Option A: Use real Bitcoin data**

Download from exchanges like Binance, Coinbase, or data providers like CryptoDataDownload.

**Option B: Generate sample data for testing**

```bash
cd scripts
python generate_sample_data.py \
    --output ../data/raw/bitcoin_sample.csv \
    --years 3 \
    --start_date 2020-01-01
```

This creates ~3 years of synthetic hourly Bitcoin data.

## Running the Training Pipeline

### Local Testing (Small Dataset)

For quick testing on your local machine:

```bash
cd scripts
python train_models.py \
    --data_path ../data/raw/bitcoin_sample.csv \
    --train_year 2020 \
    --val_year 2021 \
    --conf_year 2022 \
    --output_dir ../results \
    --n_jobs 4 \
    --verbose 2
```

### SLURM Cluster (Full Training)

1. **Edit the SLURM script** (`scripts/run_slurm.sh`):
   - Update `DATA_PATH` to point to your Bitcoin data
   - Update `SLURM_MAIL_USER` to your email
   - Adjust years if needed (TRAIN_YEAR, VAL_YEAR, CONF_YEAR)
   - Modify SLURM parameters if needed (partition, memory, etc.)

2. **Create logs directory**:
   ```bash
   mkdir -p logs
   ```

3. **Submit the job**:
   ```bash
   cd scripts
   sbatch run_slurm.sh
   ```

4. **Monitor the job**:
   ```bash
   # Check job status
   squeue -u $USER

   # Watch output in real-time
   tail -f logs/bitcoin_<jobid>.out

   # Check errors
   tail -f logs/bitcoin_<jobid>.err
   ```

## Understanding the Output

After training completes, you'll find results in `results/run_TIMESTAMP/`:

```
results/run_20231201_120000/
├── config.json                    # Configuration used
├── training/                      # Training year results
│   ├── final_best_models_*.pkl   # Best models per day
│   └── final_model_selector_*.pkl# Model selector state
├── validation/                    # Validation year results
│   └── ...
├── training_summary.json          # Training summary
└── validation_summary.json        # Validation summary
```

### Key Files

**config.json** - Full configuration of the run

**training_summary.json** - Contains:
- Best model name and parameters
- Model vote counts (how often each model won)
- Performance scores

**best_models_*.pkl** - Serialized models and predictions for each day

## Expected Runtime

With 64 cores and typical Bitcoin data:

| Model Type | Configurations | Est. Time per Day | Total Time (365 days) |
|------------|---------------|-------------------|----------------------|
| SVR Gaussian | 48 | ~10 min | ~4 days |
| Random Forest | 162 | ~2 min | ~20 hours |
| Gradient Boosting | 162 | ~4 min | ~1.5 days |
| MLP | 48 | ~6 min | ~3 days |

**Total estimated time: 7-10 days** (includes validation and confidence sets)

With 64 cores running in parallel, this reduces significantly.

## Troubleshooting

### Out of Memory

If you run out of memory:
1. Increase `#SBATCH --mem=` in `run_slurm.sh`
2. Or reduce the number of parallel jobs with `--n_jobs 32`

### Job Timeout

If job times out:
1. Increase `#SBATCH --time=` in `run_slurm.sh`
2. Or split into multiple years

### Import Errors

Make sure Python can find the modules:
```bash
export PYTHONPATH="/path/to/model_implementation:$PYTHONPATH"
```

### Data Format Issues

Check your CSV has required columns:
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.columns)
print(df.head())
```

Required: `timestamp, open, high, low, close, volume`

## Next Steps

1. **Analyze Results**: Look at `training_summary.json` to see which model performed best

2. **Visualize Predictions**: Write a script to plot predictions vs actual prices

3. **Compute Confidence Intervals**: Use the confidence year data

4. **Deploy Best Model**: Extract the best model and use it for real predictions

5. **Experiment**: Try different hyperparameters in `configs/model_configs.yaml`

## Support

- Check the main README.md for detailed documentation
- Review the PDF reference (Masters, 2018) for methodology
- Check error logs in `logs/` directory

## Tips for Best Results

1. **Data Quality**: Ensure your Bitcoin data is clean and continuous (no large gaps)

2. **Computational Resources**: More cores = faster training. Request 64+ cores if available.

3. **Hyperparameter Tuning**: Start with default configs, then refine based on results.

4. **Monitor Progress**: Check logs regularly to catch issues early.

5. **Stratification**: The code automatically handles stratification for consistency.

6. **Overlap Management**: The 1-day gap prevents future leak between datasets (24h > 23h required per Masters 2018).

## Quick Commands Reference

```bash
# Generate sample data
python scripts/generate_sample_data.py --output data/raw/bitcoin.csv --years 3

# Run locally (testing)
python scripts/train_models.py --data_path data/raw/bitcoin.csv \
    --train_year 2020 --val_year 2021 --conf_year 2022 --n_jobs 4

# Submit to SLURM
sbatch scripts/run_slurm.sh

# Check job status
squeue -u $USER

# Cancel job
scancel <jobid>

# View results
less results/run_*/training_summary.json
```

Happy predicting! 🚀📈
