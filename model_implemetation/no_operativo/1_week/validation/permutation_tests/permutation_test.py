#!/usr/bin/env python3
"""
Monte Carlo Permutation Test for Model Validation

This script implements permutation testing to validate whether a model's
performance is statistically significant or due to random chance.

Methodology:
1. Train model on real data and calculate performance metric
2. Permute the data N times (destroying temporal patterns)
3. Train model on each permutation and calculate metric
4. Calculate p-value: proportion of permutations >= real performance
5. If p < 0.05, performance is statistically significant

Based on:
- Timothy Masters "Assessing and Improving Prediction and Classification"
- neurotrader888's MCPT implementation
- White's Reality Check methodology

Usage:
    python permutation_test.py --model-path ../results/models/best_model.pkl \
                               --data ../data/processed/validation_data.csv \
                               --n-permutations 1000 \
                               --metric sharpe_ratio \
                               --output ../validation/results/permutation_test_results.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tqdm import tqdm
    import dask.bag as db
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install: pip install tqdm joblib")
    sys.exit(1)

from validation.utils import (
    permute_ohlcv_bars,
    calculate_p_value,
    calculate_percentile_rank,
    validate_predictions,
    print_validation_summary
)

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from dask_utils import get_dask_client, close_dask_client


class PermutationTest:
    """
    Monte Carlo Permutation Test for model validation.
    """

    def __init__(self,
                 n_permutations: int = 1000,
                 metric: str = 'sharpe_ratio',
                 n_jobs: int = -1,
                 random_seed: int = 42,
                 verbose: bool = True):
        """
        Initialize permutation test.

        Args:
            n_permutations: Number of permutations to run
            metric: Metric to evaluate ('sharpe_ratio', 'profit_factor', 'rmse', etc.)
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_seed: Random seed for reproducibility
            verbose: Print progress information
        """
        self.n_permutations = n_permutations
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose

        self.real_metric = None
        self.permuted_metrics = []
        self.p_value = None
        self.percentile_rank = None

    def run_single_permutation(self,
                              data: pd.DataFrame,
                              model,
                              prediction_function: Callable,
                              permutation_idx: int) -> float:
        """
        Run a single permutation test.

        Args:
            data: Input data (OHLCV DataFrame)
            model: Trained model
            prediction_function: Function that takes (model, data) and returns predictions
            permutation_idx: Index of permutation (for seeding)

        Returns:
            Metric value for this permutation
        """
        # Permute the data
        seed = self.random_seed + permutation_idx
        permuted_data = permute_ohlcv_bars(data, seed=seed)

        # Make predictions on permuted data
        try:
            predictions = prediction_function(model, permuted_data)

            # Extract actual values (assuming last column or 'close' is target)
            if 'close' in permuted_data.columns:
                actual = permuted_data['close'].values
            else:
                actual = permuted_data.iloc[:, -1].values

            # Calculate metrics
            metrics = validate_predictions(predictions, actual[:len(predictions)])

            # Return the requested metric
            metric_value = metrics.get(self.metric, 0.0)

        except Exception as e:
            if self.verbose:
                warnings.warn(f"Permutation {permutation_idx} failed: {e}")
            metric_value = 0.0

        return metric_value

    def run_test(self,
                data: pd.DataFrame,
                model,
                prediction_function: Callable,
                actual_values: np.ndarray) -> Dict[str, Any]:
        """
        Run the full permutation test.

        Args:
            data: Input data (OHLCV DataFrame)
            model: Trained model
            prediction_function: Function that takes (model, data) and returns predictions
            actual_values: Actual target values

        Returns:
            Dictionary with test results
        """
        # Calculate real metric
        if self.verbose:
            print("\nCalculating metric on real data...")

        real_predictions = prediction_function(model, data)
        real_metrics = validate_predictions(real_predictions, actual_values[:len(real_predictions)])
        self.real_metric = real_metrics.get(self.metric, 0.0)

        if self.verbose:
            print(f"Real {self.metric}: {self.real_metric:.6f}")
            print(f"\nRunning {self.n_permutations} permutations...")

        # Run permutations with Dask
        tasks = [(data, model, prediction_function, i) for i in range(self.n_permutations)]
        bag = db.from_sequence(tasks, partition_size=10)
        self.permuted_metrics = bag.map(lambda x: self.run_single_permutation(*x)).compute()

        # Calculate p-value
        # For metrics where higher is better (sharpe, profit_factor, r2)
        if self.metric in ['sharpe_ratio', 'profit_factor', 'r2', 'directional_accuracy']:
            alternative = 'greater'
        # For metrics where lower is better (mse, rmse, mae)
        elif self.metric in ['mse', 'rmse', 'mae']:
            alternative = 'less'
        else:
            alternative = 'two-sided'

        self.p_value = calculate_p_value(
            self.real_metric,
            self.permuted_metrics,
            alternative=alternative
        )

        self.percentile_rank = calculate_percentile_rank(
            self.real_metric,
            self.permuted_metrics
        )

        # Compile results
        results = {
            'test_info': {
                'n_permutations': self.n_permutations,
                'metric': self.metric,
                'random_seed': self.random_seed,
                'timestamp': datetime.now().isoformat()
            },
            'real_metric': float(self.real_metric),
            'permuted_metrics': {
                'mean': float(np.mean(self.permuted_metrics)),
                'std': float(np.std(self.permuted_metrics)),
                'min': float(np.min(self.permuted_metrics)),
                'max': float(np.max(self.permuted_metrics)),
                'median': float(np.median(self.permuted_metrics)),
                'q25': float(np.percentile(self.permuted_metrics, 25)),
                'q75': float(np.percentile(self.permuted_metrics, 75))
            },
            'p_value': float(self.p_value),
            'percentile_rank': float(self.percentile_rank),
            'is_significant': {
                'p_0.01': self.p_value < 0.01,
                'p_0.05': self.p_value < 0.05,
                'p_0.10': self.p_value < 0.10
            },
            'all_metrics': real_metrics
        }

        return results

    def plot_distribution(self, save_path: Optional[str] = None):
        """
        Plot the distribution of permuted metrics vs real metric.

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style('whitegrid')

            fig, ax = plt.subplots(figsize=(12, 6))

            # Histogram of permuted metrics
            ax.hist(self.permuted_metrics, bins=50, alpha=0.7,
                   label='Permuted Data', edgecolor='black')

            # Real metric line
            ax.axvline(self.real_metric, color='red', linestyle='--',
                      linewidth=2, label=f'Real Data ({self.metric}={self.real_metric:.4f})')

            # Critical values
            p95 = np.percentile(self.permuted_metrics, 95)
            p99 = np.percentile(self.permuted_metrics, 99)

            ax.axvline(p95, color='orange', linestyle=':',
                      linewidth=1.5, label='95th percentile')
            ax.axvline(p99, color='darkred', linestyle=':',
                      linewidth=1.5, label='99th percentile')

            ax.set_xlabel(f'{self.metric.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Monte Carlo Permutation Test\n'
                        f'p-value = {self.p_value:.4f}, '
                        f'Percentile Rank = {self.percentile_rank:.1f}%',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")

            plt.show()

        except ImportError:
            print("Warning: matplotlib/seaborn not installed. Skipping plot.")


def load_model(model_path: str):
    """Load trained model from file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def default_prediction_function(model, data: pd.DataFrame) -> np.ndarray:
    """
    Default prediction function.

    Assumes model has a .predict() method and data needs
    to be converted to feature matrix.

    Args:
        model: Trained model
        data: Input data

    Returns:
        Predictions
    """
    # Extract features (all columns except timestamp if present)
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'date']]
    X = data[feature_cols].values

    predictions = model.predict(X)

    return predictions


def main():
    client = get_dask_client()

    parser = argparse.ArgumentParser(
        description='Monte Carlo Permutation Test for Model Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pkl file)'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to validation data (CSV file with OHLCV data)'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutations (default: 1000)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='sharpe_ratio',
        choices=['sharpe_ratio', 'profit_factor', 'rmse', 'mae', 'r2', 'directional_accuracy'],
        help='Metric to evaluate (default: sharpe_ratio)'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (default: -1 for all cores)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='permutation_test_results.json',
        help='Output JSON file for results'
    )

    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Path to save distribution plot (optional)'
    )

    args = parser.parse_args()

    print("="*80)
    print("MONTE CARLO PERMUTATION TEST")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data}")
    print(f"Permutations: {args.n_permutations}")
    print(f"Metric: {args.metric}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path)
    print("Model loaded successfully")

    # Load data
    print("\nLoading data...")
    data = pd.read_csv(args.data)
    print(f"Data shape: {data.shape}")

    # Extract actual values
    if 'close' in data.columns:
        actual_values = data['close'].values
    else:
        actual_values = data.iloc[:, -1].values

    # Run permutation test
    test = PermutationTest(
        n_permutations=args.n_permutations,
        metric=args.metric,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        verbose=True
    )

    results = test.run_test(
        data=data,
        model=model,
        prediction_function=default_prediction_function,
        actual_values=actual_values
    )

    # Print results
    print_validation_summary(results['all_metrics'], title="Real Data Performance")

    print("\n" + "="*80)
    print("PERMUTATION TEST RESULTS")
    print("="*80)
    print(f"Real {args.metric}: {results['real_metric']:.6f}")
    print(f"\nPermuted {args.metric} Statistics:")
    print(f"  Mean:   {results['permuted_metrics']['mean']:.6f}")
    print(f"  Std:    {results['permuted_metrics']['std']:.6f}")
    print(f"  Median: {results['permuted_metrics']['median']:.6f}")
    print(f"  Min:    {results['permuted_metrics']['min']:.6f}")
    print(f"  Max:    {results['permuted_metrics']['max']:.6f}")
    print(f"\nStatistical Significance:")
    print(f"  p-value:         {results['p_value']:.6f}")
    print(f"  Percentile Rank: {results['percentile_rank']:.2f}%")

    if results['is_significant']['p_0.01']:
        print(f"  Significance:    *** (p < 0.01) - Highly Significant!")
    elif results['is_significant']['p_0.05']:
        print(f"  Significance:    ** (p < 0.05) - Significant")
    elif results['is_significant']['p_0.10']:
        print(f"  Significance:    * (p < 0.10) - Marginally Significant")
    else:
        print(f"  Significance:    NOT SIGNIFICANT (p >= 0.10)")

    print("="*80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Plot if requested
    if args.plot:
        test.plot_distribution(save_path=args.plot)

    close_dask_client(client)

    # Return exit code based on significance
    if results['is_significant']['p_0.05']:
        print("\n✓ Model performance is statistically significant!")
        sys.exit(0)
    else:
        print("\n✗ Model performance is NOT statistically significant.")
        print("  Performance may be due to random chance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
