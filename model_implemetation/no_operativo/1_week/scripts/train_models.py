#!/usr/bin/env python3
"""
Main Training Script for Bitcoin 1-Week Prediction Model

This script orchestrates the complete walk-forward training process:
1. Load Bitcoin hourly data
2. Split into 3 years (training, validation, confidence)
3. Run walk-forward testing on training year
4. Validate on validation year
5. Compute confidence intervals on confidence year

Usage:
    python train_models.py --data_path /path/to/bitcoin_data.csv \
                           --train_year 2020 \
                           --val_year 2021 \
                           --conf_year 2022 \
                           --output_dir ../results
"""

import argparse
import sys
import os
from datetime import datetime
import json
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import BitcoinDataLoader
from src.walk_forward import WalkForwardTester
from src.evaluation import PerformanceMetrics, ConfidenceIntervals
from src.models import ModelFactory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Bitcoin prediction models with walk-forward testing'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to CSV file with Bitcoin hourly data'
    )

    parser.add_argument(
        '--train_year',
        type=int,
        required=True,
        help='Year to use for training (e.g., 2020)'
    )

    parser.add_argument(
        '--val_year',
        type=int,
        required=True,
        help='Year to use for validation (e.g., 2021)'
    )

    parser.add_argument(
        '--conf_year',
        type=int,
        required=True,
        help='Year to use for confidence intervals (e.g., 2022)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 = all cores)'
    )

    parser.add_argument(
        '--lookback_hours',
        type=int,
        default=168,
        help='Hours of historical data to use (default: 168 = 7 days)'
    )

    parser.add_argument(
        '--lookahead_hours',
        type=int,
        default=24,
        help='Hours to predict ahead (default: 24 = 1 day)'
    )

    parser.add_argument(
        '--gap_days',
        type=int,
        default=9,
        help='Gap in days between datasets to avoid overlap (default: 9)'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0, 1, or 2)'
    )

    return parser.parse_args()


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def main():
    """Main training pipeline."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print_header("BITCOIN PRICE PREDICTION - 1 WEEK MODEL")
    print(f"Run ID: {timestamp}")
    print(f"Output directory: {run_dir}")
    print(f"\nConfiguration:")
    print(f"  Data: {args.data_path}")
    print(f"  Training year: {args.train_year}")
    print(f"  Validation year: {args.val_year}")
    print(f"  Confidence year: {args.conf_year}")
    print(f"  Lookback: {args.lookback_hours} hours")
    print(f"  Lookahead: {args.lookahead_hours} hours")
    print(f"  Gap: {args.gap_days} days")
    print(f"  Parallel jobs: {args.n_jobs}")

    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # STEP 1: LOAD AND SPLIT DATA
    # ========================================================================
    print_header("STEP 1: LOAD AND SPLIT DATA")

    loader = BitcoinDataLoader(
        lookback_hours=args.lookback_hours,
        lookahead_hours=args.lookahead_hours,
        gap_days=args.gap_days
    )

    print("Loading Bitcoin data...")
    df_full = loader.load_data(args.data_path)
    print(f"  Total data points: {len(df_full)}")
    print(f"  Date range: {df_full.index.min()} to {df_full.index.max()}")

    print("\nSplitting into years...")
    df_train, df_val, df_conf = loader.split_into_years(
        df_full,
        train_year=args.train_year,
        val_year=args.val_year,
        conf_year=args.conf_year
    )

    # ========================================================================
    # STEP 2: WALK-FORWARD TRAINING (YEAR 1)
    # ========================================================================
    print_header("STEP 2: WALK-FORWARD TRAINING (TRAINING YEAR)")

    tester = WalkForwardTester(
        lookback_hours=args.lookback_hours,
        lookahead_hours=args.lookahead_hours,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )

    train_summary = tester.run_year(
        df_train,
        save_dir=os.path.join(run_dir, 'training')
    )

    # Save training summary
    with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        summary_json = {
            'vote_counts': train_summary['vote_counts'],
            'best_model': train_summary['best_model'],
            'best_params': train_summary['best_params'],
            'best_score': float(train_summary['best_score'])
        }
        json.dump(summary_json, f, indent=2)

    # ========================================================================
    # STEP 3: VALIDATION (YEAR 2)
    # ========================================================================
    print_header("STEP 3: VALIDATION (VALIDATION YEAR)")

    print("Using best model from training year:")
    print(f"  Model: {train_summary['best_model']}")
    print(f"  Parameters: {train_summary['best_params']}")

    # Create best model
    best_model = ModelFactory.create_multioutput_model(
        train_summary['best_model'],
        train_summary['best_params']
    )

    # Run validation
    print("\nRunning walk-forward validation...")
    val_tester = WalkForwardTester(
        lookback_hours=args.lookback_hours,
        lookahead_hours=args.lookahead_hours,
        n_jobs=1,  # Only one model, no need for parallel
        verbose=args.verbose
    )

    val_summary = val_tester.run_year(
        df_val,
        save_dir=os.path.join(run_dir, 'validation')
    )

    # Save validation summary
    with open(os.path.join(run_dir, 'validation_summary.json'), 'w') as f:
        summary_json = {
            'best_score': float(val_summary['best_score'])
        }
        json.dump(summary_json, f, indent=2)

    # ========================================================================
    # STEP 4: CONFIDENCE INTERVALS (YEAR 3)
    # ========================================================================
    print_header("STEP 4: CONFIDENCE INTERVALS (CONFIDENCE YEAR)")

    print("Computing confidence intervals using Year 3...")
    print("(Not fully implemented - placeholder)")

    # This would involve:
    # 1. Run best model on confidence year
    # 2. Collect all prediction errors
    # 3. Compute empirical quantiles
    # 4. Compute confidence bounds

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_header("TRAINING COMPLETE")

    print("Results saved to:", run_dir)
    print("\nBest Model:")
    print(f"  Name: {train_summary['best_model']}")
    print(f"  Training MSE: {train_summary['best_score']:.6f}")
    print(f"  Validation MSE: {val_summary['best_score']:.6f}")

    print("\nModel Vote Counts (Training Year):")
    for model, count in sorted(train_summary['vote_counts'].items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {model:25s}: {count:4d} days")

    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
