#!/bin/bash
# Example script to run complete validation pipeline

echo "========================================="
echo "Complete Validation Pipeline Example"
echo "========================================="

# Configuration
DATA_FILE="../data/raw/bitcoin_hourly.csv"
MODEL_TYPE="RandomForestRegressor"
N_PERMUTATIONS=1000
METRIC="sharpe_ratio"
OUTPUT_DIR="results/full_validation/"

# Check if data exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Please download data first using:"
    echo "  cd ../scripts"
    echo "  python download_bitcoin_data.py"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Data: $DATA_FILE"
echo "  Model: $MODEL_TYPE"
echo "  Permutations: $N_PERMUTATIONS"
echo "  Metric: $METRIC"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting validation pipeline..."
echo ""

# Run full validation pipeline
python full_validation_pipeline.py \
    --data "$DATA_FILE" \
    --model-type "$MODEL_TYPE" \
    --n-permutations "$N_PERMUTATIONS" \
    --metric "$METRIC" \
    --output-dir "$OUTPUT_DIR" \
    --n-jobs -1 \
    --seed 42

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Validation PASSED - Model is statistically significant!"
    echo ""
else
    echo ""
    echo "✗ Validation FAILED - Model performance likely due to luck"
    echo ""
    exit 1
fi

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review results JSON file"
echo "  2. If passed, proceed to production testing"
echo "  3. If failed, try different features/models"
