#!/usr/bin/env python3
"""
Complete Validation Pipeline - In-Sample & Walk-Forward Permutation Tests

This script implements the complete validation framework:
1. In-Sample Excellence Test - Does the model perform well on training data?
2. In-Sample Permutation Test - Is in-sample performance significant? (p-value)
3. Walk-Forward Test - Does the model perform well out-of-sample?
4. Walk-Forward Permutation Test - Is walk-forward performance significant? (p-value)

Only if ALL tests pass, the model is considered valid and not due to luck.

Based on:
- Timothy Masters "Assessing and Improving Prediction and Classification"
- neurotrader888's MCPT framework
- Masters' methodology for eliminating luck-based results

Usage:
    python full_validation_pipeline.py \
        --data ../data/raw/bitcoin_hourly.csv \
        --model-type RandomForestRegressor \
        --n-permutations 1000 \
        --output-dir results/full_validation/
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable, Optional
from datetime import datetime
import warnings

sys.path.append(str(Path(__file__).parent.parent))

try:
    from tqdm import tqdm
    from dask.distributed import Client
    import dask.bag as db
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install: pip install scikit-learn tqdm joblib")
    sys.exit(1)

from validation.utils import (
    permute_ohlcv_bars,
    calculate_p_value,
    calculate_percentile_rank,
    calculate_sharpe_ratio,
    calculate_profit_factor,
    print_validation_summary
)

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from dask_utils import get_dask_client, close_dask_client


class FullValidationPipeline:
    """
    Complete validation pipeline with all 4 tests.
    """

    def __init__(self,
                 model_class,
                 model_params: Dict[str, Any],
                 n_permutations: int = 1000,
                 walk_forward_window: int = 168,  # 7 days
                 walk_forward_step: int = 24,     # 1 day
                 prediction_horizon: int = 24,     # 1 day
                 metric: str = 'sharpe_ratio',
                 n_jobs: int = -1,
                 random_seed: int = 42,
                 verbose: bool = True):
        """
        Initialize validation pipeline.

        Args:
            model_class: Sklearn model class (e.g., RandomForestRegressor)
            model_params: Parameters for the model
            n_permutations: Number of permutations for each test
            walk_forward_window: Input window size (hours)
            walk_forward_step: Step size for walk-forward (hours)
            prediction_horizon: Prediction horizon (hours)
            metric: Metric to evaluate
            n_jobs: Number of parallel jobs
            random_seed: Random seed
            verbose: Print progress
        """
        self.model_class = model_class
        self.model_params = model_params
        self.n_permutations = n_permutations
        self.walk_forward_window = walk_forward_window
        self.walk_forward_step = walk_forward_step
        self.prediction_horizon = prediction_horizon
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.results = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (features, targets)
        """
        # Use OHLCV as features
        feature_cols = ['open', 'high', 'low', 'close', 'volume']

        # Create sequences
        X_list = []
        y_list = []

        for i in range(len(df) - self.walk_forward_window - self.prediction_horizon):
            # Input: window of OHLCV
            X_window = df[feature_cols].iloc[i:i+self.walk_forward_window].values.flatten()

            # Target: mean close price over prediction horizon
            y_target = df['close'].iloc[
                i+self.walk_forward_window:i+self.walk_forward_window+self.prediction_horizon
            ].mean()

            X_list.append(X_window)
            y_list.append(y_target)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def calculate_trading_metric(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate trading performance metric.

        Args:
            predictions: Predicted values
            actual: Actual values

        Returns:
            Metric value
        """
        if self.metric == 'sharpe_ratio':
            # Generate trading returns
            returns = []
            for i in range(len(predictions) - 1):
                if predictions[i+1] > predictions[i]:  # Predict up -> long
                    ret = (actual[i+1] - actual[i]) / actual[i]
                else:
                    ret = 0
                returns.append(ret)

            return calculate_sharpe_ratio(np.array(returns))

        elif self.metric == 'profit_factor':
            return calculate_profit_factor(predictions, actual)

        elif self.metric == 'rmse':
            return -np.sqrt(mean_squared_error(actual, predictions))  # Negative for minimization

        elif self.metric == 'r2':
            return r2_score(actual, predictions)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ==================== TEST 1: IN-SAMPLE EXCELLENCE ====================

    def test_1_insample_excellence(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Test 1: In-Sample Excellence Test

        Train model on in-sample data and evaluate performance.
        This establishes baseline - does the model work at all?

        Args:
            df_train: Training data

        Returns:
            Dictionary with test results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 1: IN-SAMPLE EXCELLENCE TEST")
            print("="*80)

        # Prepare data
        X_train, y_train = self.prepare_features(df_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        if self.verbose:
            print("Training model on in-sample data...")

        model = self.model_class(**self.model_params, random_state=self.random_seed)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        predictions = model.predict(X_train_scaled)

        # Calculate metric
        metric_value = self.calculate_trading_metric(predictions, y_train)

        # Calculate additional metrics
        mse = mean_squared_error(y_train, predictions)
        r2 = r2_score(y_train, predictions)

        results = {
            'test_name': 'in_sample_excellence',
            'metric': self.metric,
            'metric_value': float(metric_value),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'n_samples': len(y_train),
            'passes': metric_value > 0 if self.metric in ['sharpe_ratio', 'profit_factor', 'r2'] else True
        }

        if self.verbose:
            print(f"\nIn-Sample Performance:")
            print(f"  {self.metric}: {metric_value:.6f}")
            print(f"  RMSE: {np.sqrt(mse):.6f}")
            print(f"  R²: {r2:.6f}")
            print(f"  Status: {'✓ PASS' if results['passes'] else '✗ FAIL'}")

        self.trained_model = model
        return results

    # ==================== TEST 2: IN-SAMPLE PERMUTATION ====================

    def test_2_insample_permutation(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Test 2: In-Sample Permutation Test

        Permute training data and check if real performance is significant.
        This tests if in-sample performance is due to real patterns vs overfitting to noise.

        Args:
            df_train: Training data

        Returns:
            Dictionary with test results including p-value
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 2: IN-SAMPLE PERMUTATION TEST")
            print("="*80)

        # Get real metric from Test 1
        real_metric = self.results['test_1']['metric_value']

        if self.verbose:
            print(f"Real in-sample {self.metric}: {real_metric:.6f}")
            print(f"Running {self.n_permutations} permutations...")

        # Run permutations with Dask
        tasks = [(df_train, i) for i in range(self.n_permutations)]
        bag = db.from_sequence(tasks, partition_size=10)
        permuted_metrics = bag.map(lambda x: self._run_insample_permutation(*x)).compute()

        # Calculate p-value
        if self.metric in ['sharpe_ratio', 'profit_factor', 'r2']:
            alternative = 'greater'
        else:
            alternative = 'less'

        p_value = calculate_p_value(real_metric, permuted_metrics, alternative)
        percentile = calculate_percentile_rank(real_metric, permuted_metrics)

        results = {
            'test_name': 'in_sample_permutation',
            'real_metric': float(real_metric),
            'permuted_mean': float(np.mean(permuted_metrics)),
            'permuted_std': float(np.std(permuted_metrics)),
            'permuted_median': float(np.median(permuted_metrics)),
            'p_value': float(p_value),
            'percentile_rank': float(percentile),
            'n_permutations': self.n_permutations,
            'passes_p01': p_value < 0.01,
            'passes_p05': p_value < 0.05,
            'passes_p10': p_value < 0.10
        }

        if self.verbose:
            print(f"\nPermutation Test Results:")
            print(f"  Real metric:      {real_metric:.6f}")
            print(f"  Permuted mean:    {results['permuted_mean']:.6f}")
            print(f"  Permuted std:     {results['permuted_std']:.6f}")
            print(f"  p-value:          {p_value:.6f}")
            print(f"  Percentile rank:  {percentile:.2f}%")

            if p_value < 0.01:
                print(f"  Status: ✓✓✓ HIGHLY SIGNIFICANT (p < 0.01)")
            elif p_value < 0.05:
                print(f"  Status: ✓✓ SIGNIFICANT (p < 0.05)")
            elif p_value < 0.10:
                print(f"  Status: ✓ MARGINALLY SIGNIFICANT (p < 0.10)")
            else:
                print(f"  Status: ✗ NOT SIGNIFICANT (p >= 0.10)")

        return results

    def _run_insample_permutation(self, df_train: pd.DataFrame, perm_idx: int) -> float:
        """Run single in-sample permutation."""
        # Permute data
        seed = self.random_seed + perm_idx
        df_permuted = permute_ohlcv_bars(df_train, seed=seed)

        # Prepare features
        X_perm, y_perm = self.prepare_features(df_permuted)
        X_perm_scaled = StandardScaler().fit_transform(X_perm)

        # Train model
        model = self.model_class(**self.model_params, random_state=seed)
        try:
            model.fit(X_perm_scaled, y_perm)
            predictions = model.predict(X_perm_scaled)
            metric_value = self.calculate_trading_metric(predictions, y_perm)
        except:
            metric_value = 0.0

        return metric_value

    # ==================== TEST 3: WALK-FORWARD ====================

    def test_3_walk_forward(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Test 3: Walk-Forward Test

        Evaluate model on out-of-sample data using walk-forward methodology.
        This tests if the model generalizes to unseen data.

        Args:
            df_test: Test data (validation set)

        Returns:
            Dictionary with test results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 3: WALK-FORWARD TEST")
            print("="*80)

        # Prepare data
        X_test, y_test = self.prepare_features(df_test)
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions using trained model from Test 1
        if self.verbose:
            print("Making walk-forward predictions...")

        predictions = self.trained_model.predict(X_test_scaled)

        # Calculate metric
        metric_value = self.calculate_trading_metric(predictions, y_test)

        # Additional metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results = {
            'test_name': 'walk_forward',
            'metric': self.metric,
            'metric_value': float(metric_value),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'n_samples': len(y_test),
            'passes': metric_value > 0 if self.metric in ['sharpe_ratio', 'profit_factor', 'r2'] else True
        }

        if self.verbose:
            print(f"\nWalk-Forward Performance:")
            print(f"  {self.metric}: {metric_value:.6f}")
            print(f"  RMSE: {np.sqrt(mse):.6f}")
            print(f"  R²: {r2:.6f}")
            print(f"  Status: {'✓ PASS' if results['passes'] else '✗ FAIL'}")

        self.test_predictions = predictions
        self.test_actual = y_test

        return results

    # ==================== TEST 4: WALK-FORWARD PERMUTATION ====================

    def test_4_walk_forward_permutation(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Test 4: Walk-Forward Permutation Test

        Permute test data and check if real walk-forward performance is significant.
        This is THE critical test - if this fails, performance is due to LUCK.

        Args:
            df_test: Test data

        Returns:
            Dictionary with test results including p-value
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 4: WALK-FORWARD PERMUTATION TEST")
            print("="*80)
            print("This is the CRITICAL test - determines if results are due to luck")

        # Get real metric from Test 3
        real_metric = self.results['test_3']['metric_value']

        if self.verbose:
            print(f"Real walk-forward {self.metric}: {real_metric:.6f}")
            print(f"Running {self.n_permutations} permutations...")

        # Run permutations with Dask
        tasks = [(df_test, i) for i in range(self.n_permutations)]
        bag = db.from_sequence(tasks, partition_size=10)
        permuted_metrics = bag.map(lambda x: self._run_walkforward_permutation(*x)).compute()

        # Calculate p-value
        if self.metric in ['sharpe_ratio', 'profit_factor', 'r2']:
            alternative = 'greater'
        else:
            alternative = 'less'

        p_value = calculate_p_value(real_metric, permuted_metrics, alternative)
        percentile = calculate_percentile_rank(real_metric, permuted_metrics)

        results = {
            'test_name': 'walk_forward_permutation',
            'real_metric': float(real_metric),
            'permuted_mean': float(np.mean(permuted_metrics)),
            'permuted_std': float(np.std(permuted_metrics)),
            'permuted_median': float(np.median(permuted_metrics)),
            'permuted_min': float(np.min(permuted_metrics)),
            'permuted_max': float(np.max(permuted_metrics)),
            'p_value': float(p_value),
            'percentile_rank': float(percentile),
            'n_permutations': self.n_permutations,
            'passes_p01': p_value < 0.01,
            'passes_p05': p_value < 0.05,
            'passes_p10': p_value < 0.10
        }

        if self.verbose:
            print(f"\nWalk-Forward Permutation Results:")
            print(f"  Real metric:      {real_metric:.6f}")
            print(f"  Permuted mean:    {results['permuted_mean']:.6f}")
            print(f"  Permuted std:     {results['permuted_std']:.6f}")
            print(f"  Permuted range:   [{results['permuted_min']:.6f}, {results['permuted_max']:.6f}]")
            print(f"  p-value:          {p_value:.6f}")
            print(f"  Percentile rank:  {percentile:.2f}%")

            if p_value < 0.01:
                print(f"  Status: ✓✓✓ HIGHLY SIGNIFICANT (p < 0.01) - NOT LUCK!")
            elif p_value < 0.05:
                print(f"  Status: ✓✓ SIGNIFICANT (p < 0.05) - Probably not luck")
            elif p_value < 0.10:
                print(f"  Status: ✓ MARGINALLY SIGNIFICANT (p < 0.10) - Weak evidence")
            else:
                print(f"  Status: ✗ NOT SIGNIFICANT (p >= 0.10) - LIKELY LUCK!")

        return results

    def _run_walkforward_permutation(self, df_test: pd.DataFrame, perm_idx: int) -> float:
        """Run single walk-forward permutation."""
        # Permute test data
        seed = self.random_seed + perm_idx + 10000  # Different seed space
        df_permuted = permute_ohlcv_bars(df_test, seed=seed)

        # Prepare features
        X_perm, y_perm = self.prepare_features(df_permuted)
        X_perm_scaled = self.scaler.transform(X_perm)

        # Use trained model (from Test 1) on permuted data
        try:
            predictions = self.trained_model.predict(X_perm_scaled)
            metric_value = self.calculate_trading_metric(predictions, y_perm)
        except:
            metric_value = 0.0

        return metric_value

    # ==================== MAIN PIPELINE ====================

    def run_full_pipeline(self,
                         df_train: pd.DataFrame,
                         df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete 4-test validation pipeline.

        Args:
            df_train: Training data (in-sample)
            df_test: Test data (out-of-sample / walk-forward)

        Returns:
            Complete results dictionary
        """
        start_time = datetime.now()

        if self.verbose:
            print("\n" + "="*80)
            print("COMPLETE VALIDATION PIPELINE")
            print("="*80)
            print(f"Model: {self.model_class.__name__}")
            print(f"Metric: {self.metric}")
            print(f"Permutations: {self.n_permutations}")
            print(f"Training samples: {len(df_train)}")
            print(f"Test samples: {len(df_test)}")
            print("="*80)

        # Test 1: In-Sample Excellence
        self.results['test_1'] = self.test_1_insample_excellence(df_train)

        # Test 2: In-Sample Permutation
        self.results['test_2'] = self.test_2_insample_permutation(df_train)

        # Test 3: Walk-Forward
        self.results['test_3'] = self.test_3_walk_forward(df_test)

        # Test 4: Walk-Forward Permutation (CRITICAL)
        self.results['test_4'] = self.test_4_walk_forward_permutation(df_test)

        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        final_results = {
            'pipeline_info': {
                'model_class': self.model_class.__name__,
                'model_params': self.model_params,
                'metric': self.metric,
                'n_permutations': self.n_permutations,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration
            },
            'test_1_insample_excellence': self.results['test_1'],
            'test_2_insample_permutation': self.results['test_2'],
            'test_3_walk_forward': self.results['test_3'],
            'test_4_walk_forward_permutation': self.results['test_4'],
            'final_verdict': self._generate_verdict()
        }

        if self.verbose:
            self._print_final_summary()

        return final_results

    def _generate_verdict(self) -> Dict[str, Any]:
        """Generate final verdict on model validity."""
        # Criteria for passing
        test1_pass = self.results['test_1']['passes']
        test2_pass_p05 = self.results['test_2']['passes_p05']
        test2_pass_p01 = self.results['test_2']['passes_p01']
        test3_pass = self.results['test_3']['passes']
        test4_pass_p05 = self.results['test_4']['passes_p05']
        test4_pass_p01 = self.results['test_4']['passes_p01']

        # Overall verdict
        all_pass_strict = test1_pass and test2_pass_p05 and test3_pass and test4_pass_p05
        all_pass_very_strict = test1_pass and test2_pass_p01 and test3_pass and test4_pass_p01

        verdict = {
            'test_1_pass': test1_pass,
            'test_2_pass_p05': test2_pass_p05,
            'test_2_pass_p01': test2_pass_p01,
            'test_3_pass': test3_pass,
            'test_4_pass_p05': test4_pass_p05,
            'test_4_pass_p01': test4_pass_p01,
            'overall_pass': all_pass_strict,
            'overall_pass_strict': all_pass_very_strict,
            'recommendation': ''
        }

        if all_pass_very_strict:
            verdict['recommendation'] = "EXCELLENT - Model is highly significant and not due to luck. Safe for production."
        elif all_pass_strict:
            verdict['recommendation'] = "GOOD - Model is significant and likely not due to luck. Consider for production."
        elif test4_pass_p05:
            verdict['recommendation'] = "MARGINAL - Walk-forward is significant but check other metrics. Use with caution."
        else:
            verdict['recommendation'] = "REJECT - Performance likely due to luck or overfitting. Do NOT use in production."

        return verdict

    def _print_final_summary(self):
        """Print formatted final summary."""
        print("\n" + "="*80)
        print("FINAL VALIDATION SUMMARY")
        print("="*80)

        verdict = self.results.get('final_verdict', self._generate_verdict())

        print("\nTest Results:")
        print(f"  Test 1 (In-Sample Excellence):        {'✓ PASS' if verdict['test_1_pass'] else '✗ FAIL'}")
        print(f"  Test 2 (In-Sample Permutation p<0.05): {'✓ PASS' if verdict['test_2_pass_p05'] else '✗ FAIL'}")
        print(f"  Test 2 (In-Sample Permutation p<0.01): {'✓ PASS' if verdict['test_2_pass_p01'] else '✗ FAIL'}")
        print(f"  Test 3 (Walk-Forward):                 {'✓ PASS' if verdict['test_3_pass'] else '✗ FAIL'}")
        print(f"  Test 4 (Walk-Forward Perm p<0.05):     {'✓ PASS' if verdict['test_4_pass_p05'] else '✗ FAIL'}")
        print(f"  Test 4 (Walk-Forward Perm p<0.01):     {'✓ PASS' if verdict['test_4_pass_p01'] else '✗ FAIL'}")

        print(f"\nOverall Status:")
        if verdict['overall_pass_strict']:
            print(f"  ✓✓✓ ALL TESTS PASSED (strict)")
        elif verdict['overall_pass']:
            print(f"  ✓✓ PASSED (standard)")
        else:
            print(f"  ✗ FAILED")

        print(f"\nRecommendation:")
        print(f"  {verdict['recommendation']}")

        print("\nKey p-values:")
        print(f"  In-Sample Permutation:   {self.results['test_2']['p_value']:.6f}")
        print(f"  Walk-Forward Permutation: {self.results['test_4']['p_value']:.6f}")

        print("="*80)


def load_and_split_data(data_path: str,
                       train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and split into train/test."""
    df = pd.read_csv(data_path)

    # Convert timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    # Split
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    return df_train, df_test


def main():
    # Start Dask
    client = get_dask_client()

    parser = argparse.ArgumentParser(
        description='Complete 4-Test Validation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--model-type', type=str, default='RandomForestRegressor',
                       choices=['RandomForestRegressor', 'GradientBoostingRegressor', 'SVR', 'MLPRegressor'])
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'profit_factor', 'rmse', 'r2'])
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/full_validation/')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df_train, df_test = load_and_split_data(args.data, args.train_ratio)

    # Setup model
    model_classes = {
        'RandomForestRegressor': (RandomForestRegressor, {'n_estimators': 100, 'max_depth': 10}),
        'GradientBoostingRegressor': (GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 5}),
        'SVR': (SVR, {'kernel': 'rbf', 'C': 1.0}),
        'MLPRegressor': (MLPRegressor, {'hidden_layer_sizes': (100,), 'max_iter': 500})
    }

    model_class, model_params = model_classes[args.model_type]

    # Run pipeline
    pipeline = FullValidationPipeline(
        model_class=model_class,
        model_params=model_params,
        n_permutations=args.n_permutations,
        metric=args.metric,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        verbose=True
    )

    results = pipeline.run_full_pipeline(df_train, df_test)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'validation_results_{args.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    close_dask_client(client)

    # Exit with appropriate code
    if results['final_verdict']['overall_pass']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
