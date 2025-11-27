#!/bin/bash
#SBATCH --job-name=bitcoin_1week          # Job name
#SBATCH --output=logs/bitcoin_%j.out      # Standard output log (%j = job ID)
#SBATCH --error=logs/bitcoin_%j.err       # Standard error log
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=64                # Number of CPU cores per task
#SBATCH --mem=64G                         # Memory per node
#SBATCH --time=72:00:00                   # Time limit (72 hours)
#SBATCH --partition=normal                # Partition name (adjust as needed)
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=your.email@domain.com # Your email (CHANGE THIS!)

################################################################################
# Bitcoin Price Prediction - 1 Week Model
# SLURM Batch Script for Walk-Forward Training
#
# This script runs the complete training pipeline:
# - Year 1: Walk-forward training with model selection
# - Year 2: Validation with best model
# - Year 3: Confidence interval computation
#
# Requirements:
# - Python 3.7+
# - scikit-learn, pandas, numpy, scipy, joblib
# - Bitcoin hourly candlestick data (3 years)
#
# Usage:
#   sbatch run_slurm.sh
################################################################################

echo "================================================================"
echo "Bitcoin Price Prediction - 1 Week Model"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "================================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster)
# Uncomment and modify as needed for your cluster
# module purge
# module load python/3.9
# module load gcc/9.3.0
# module load openblas/0.3.10

# Activate Python virtual environment (if using one)
# Uncomment and modify path as needed
# source /path/to/your/venv/bin/activate

# Or use conda environment
# Uncomment and modify as needed
# module load anaconda3
# source activate bitcoin_pred

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}/..:${PYTHONPATH}"

# Set number of threads for numpy/scikit-learn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print Python and library versions
echo ""
echo "Python Environment:"
echo "----------------------------------------------------------------"
python --version
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
echo "----------------------------------------------------------------"

# ======================================================================
# CONFIGURATION - MODIFY THESE PATHS AND PARAMETERS
# ======================================================================

# Path to Bitcoin data (CHANGE THIS!)
DATA_PATH="/path/to/your/bitcoin_hourly_data.csv"

# Years to use for training/validation/confidence
TRAIN_YEAR=2020
VAL_YEAR=2021
CONF_YEAR=2022

# Output directory
OUTPUT_DIR="../results"

# Model parameters
LOOKBACK_HOURS=168  # 7 days
LOOKAHEAD_HOURS=24  # 1 day
GAP_DAYS=9          # Gap between datasets

# Number of parallel jobs (use all allocated cores)
N_JOBS=$SLURM_CPUS_PER_TASK

# Verbosity
VERBOSE=2

# ======================================================================
# CHECK DATA FILE EXISTS
# ======================================================================

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Please update DATA_PATH in this script"
    exit 1
fi

echo ""
echo "Configuration:"
echo "----------------------------------------------------------------"
echo "Data file: $DATA_PATH"
echo "Training year: $TRAIN_YEAR"
echo "Validation year: $VAL_YEAR"
echo "Confidence year: $CONF_YEAR"
echo "Lookback: $LOOKBACK_HOURS hours"
echo "Lookahead: $LOOKAHEAD_HOURS hours"
echo "Gap: $GAP_DAYS days"
echo "Parallel jobs: $N_JOBS"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================"
echo ""

# ======================================================================
# RUN TRAINING SCRIPT
# ======================================================================

python train_models.py \
    --data_path "$DATA_PATH" \
    --train_year $TRAIN_YEAR \
    --val_year $VAL_YEAR \
    --conf_year $CONF_YEAR \
    --output_dir "$OUTPUT_DIR" \
    --lookback_hours $LOOKBACK_HOURS \
    --lookahead_hours $LOOKAHEAD_HOURS \
    --gap_days $GAP_DAYS \
    --n_jobs $N_JOBS \
    --verbose $VERBOSE

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================================"

# Print some job statistics
if command -v sacct &> /dev/null; then
    echo ""
    echo "Job Statistics:"
    echo "----------------------------------------------------------------"
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,Elapsed,MaxRSS,State
    echo "================================================================"
fi

exit $EXIT_CODE
