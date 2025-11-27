"""
Walk-Forward Testing Implementation

Implements walk-forward testing as described in Masters (2018), Chapter 1, pages 14-20.

This module orchestrates the daily walk-forward training and testing process,
ensuring proper overlap management and parallel model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from joblib import Parallel, delayed
import pickle
import os

from .models import ModelFactory, ModelSelector
from .feature_engineering import FeatureEngineer, prepare_sequences_for_ml
from .evaluation import PerformanceMetrics


class WalkForwardTester:
    """
    Manages walk-forward testing for time series prediction.

    Performs daily walk-forward training where each day:
    1. Use previous 168 hours (7 days) as training data
    2. Predict next 24 hours
    3. Move forward 1 day and repeat

    This follows Masters (2018) recommendations for time series validation.
    """

    def __init__(self, lookback_hours=168, lookahead_hours=24,
                 n_jobs=-1, verbose=1):
        """
        Initialize walk-forward tester.

        Args:
            lookback_hours: Hours of historical data to use (default: 168 = 7 days)
            lookahead_hours: Hours to predict ahead (default: 24 = 1 day)
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Verbosity level
        """
        self.lookback_hours = lookback_hours
        self.lookahead_hours = lookahead_hours
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.feature_engineer = FeatureEngineer(scaler_type='robust')
        self.model_selector = ModelSelector()

        self.results = []
        self.best_models_per_day = {}

    def prepare_day_data(self, df: pd.DataFrame, day_idx: int) -> Tuple:
        """
        Prepare training and test data for a single day.

        Args:
            df: Full dataframe with features
            day_idx: Index of the day (in hours from start)

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, date)
        """
        # Training window: previous lookback_hours
        train_start = day_idx - self.lookback_hours
        train_end = day_idx

        # Test window: next lookahead_hours
        test_start = day_idx
        test_end = day_idx + self.lookahead_hours

        # Check bounds
        if train_start < 0 or test_end > len(df):
            raise ValueError("Day index out of bounds")

        # Extract training data
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Create sequences (for now, just use raw features)
        X_train = train_df.values.reshape(1, -1, train_df.shape[1])
        y_train = train_df['close'].values.reshape(1, -1)

        X_test = test_df.values.reshape(1, -1, test_df.shape[1])
        y_test = test_df['close'].values.reshape(1, -1)

        # Flatten for sklearn models
        X_train_flat, y_train_flat = prepare_sequences_for_ml(X_train, y_train)
        X_test_flat, y_test_flat = prepare_sequences_for_ml(X_test, y_test)

        date = df.index[day_idx]

        return X_train_flat, y_train_flat, X_test_flat, y_test_flat, date

    def train_single_model_config(self, model_name: str, params: Dict,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  date: str) -> Dict:
        """
        Train and evaluate a single model configuration.

        Args:
            model_name: Name of model
            params: Hyperparameters
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            date: Date string

        Returns:
            Dictionary with results
        """
        try:
            # Create and train model
            model = ModelFactory.create_multioutput_model(model_name, params)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = PerformanceMetrics.mse(y_test.flatten(), y_pred.flatten())
            rmse = PerformanceMetrics.rmse(y_test.flatten(), y_pred.flatten())
            mae = PerformanceMetrics.mae(y_test.flatten(), y_pred.flatten())

            result = {
                'model_name': model_name,
                'params': params,
                'date': str(date),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'predictions': y_pred,
                'actuals': y_test,
                'success': True
            }

            return result

        except Exception as e:
            return {
                'model_name': model_name,
                'params': params,
                'date': str(date),
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'error': str(e),
                'success': False
            }

    def run_single_day(self, df_features: pd.DataFrame, day_idx: int) -> List[Dict]:
        """
        Run walk-forward test for a single day with all model configurations.

        Args:
            df_features: DataFrame with all features
            day_idx: Index of the day

        Returns:
            List of results for all models
        """
        # Prepare data for this day
        X_train, y_train, X_test, y_test, date = self.prepare_day_data(
            df_features, day_idx
        )

        # Fit scaler on training data only (important!)
        self.feature_engineer.fit_scaler(X_train)
        X_train_scaled = self.feature_engineer.transform(X_train)
        X_test_scaled = self.feature_engineer.transform(X_test)

        # Get all model configurations
        model_configs = ModelFactory.get_model_configs()

        # Generate all hyperparameter combinations
        tasks = []
        for model_name, config in model_configs.items():
            param_grid = config['param_grid']

            # Generate all combinations
            from itertools import product
            keys = param_grid.keys()
            values = param_grid.values()

            for param_values in product(*values):
                params = dict(zip(keys, param_values))
                tasks.append((model_name, params))

        # Train all models in parallel
        if self.verbose > 0:
            print(f"\n[{date}] Training {len(tasks)} model configurations...")

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self.train_single_model_config)(
                model_name, params,
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                date
            )
            for model_name, params in tasks
        )

        # Find best model for this day
        successful_results = [r for r in results if r['success']]

        if successful_results:
            best_result = min(successful_results, key=lambda x: x['mse'])
            self.best_models_per_day[str(date)] = best_result

            # Update model selector
            self.model_selector.update(
                date=str(date),
                model_name=best_result['model_name'],
                params=best_result['params'],
                score=best_result['mse'],
                predictions=best_result['predictions']
            )

            if self.verbose > 0:
                print(f"  Best: {best_result['model_name']} "
                      f"(MSE: {best_result['mse']:.4f}, "
                      f"RMSE: {best_result['rmse']:.4f})")

        return results

    def run_year(self, df_raw: pd.DataFrame, save_dir: str = None) -> Dict:
        """
        Run walk-forward testing for an entire year.

        Args:
            df_raw: DataFrame with raw OHLCV data
            save_dir: Directory to save results (optional)

        Returns:
            Dictionary with summary results
        """
        print("=" * 80)
        print("WALK-FORWARD TESTING - TRAINING YEAR")
        print("=" * 80)

        # Create features
        print("\nCreating features...")
        df_features = self.feature_engineer.create_features(df_raw)
        print(f"  Total features: {df_features.shape[1]}")

        # Calculate valid day indices (walk-forward daily)
        # Start when we have enough history
        start_idx = self.lookback_hours

        # End when we can't predict full lookahead
        end_idx = len(df_features) - self.lookahead_hours

        # Generate indices for each day (every 24 hours)
        day_indices = list(range(start_idx, end_idx, 24))

        print(f"\nWalk-forward schedule:")
        print(f"  Start date: {df_features.index[start_idx]}")
        print(f"  End date: {df_features.index[end_idx - 24]}")
        print(f"  Total days: {len(day_indices)}")

        # Run walk-forward for each day
        all_results = []

        for i, day_idx in enumerate(day_indices):
            if self.verbose > 0:
                print(f"\n[Day {i+1}/{len(day_indices)}]", end=" ")

            day_results = self.run_single_day(df_features, day_idx)
            all_results.extend(day_results)

            # Save intermediate results periodically
            if save_dir and (i + 1) % 30 == 0:  # Save every 30 days
                self.save_results(save_dir, intermediate=True)

        # Save final results
        if save_dir:
            self.save_results(save_dir, intermediate=False)

        # Generate summary
        summary = self.generate_summary(all_results)

        return summary

    def save_results(self, save_dir: str, intermediate: bool = False):
        """Save results to disk."""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "intermediate" if intermediate else "final"

        # Save best models per day
        filepath = os.path.join(save_dir, f"{prefix}_best_models_{timestamp}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_models_per_day, f)

        # Save model selector state
        filepath = os.path.join(save_dir, f"{prefix}_model_selector_{timestamp}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_selector, f)

        print(f"\n  Results saved to {save_dir}")

    def generate_summary(self, all_results: List[Dict]) -> Dict:
        """
        Generate summary statistics from all results.

        Args:
            all_results: List of all result dictionaries

        Returns:
            Summary dictionary
        """
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Get model vote counts
        vote_counts = self.model_selector.get_model_vote_counts()
        print("\nModel Selection Frequency:")
        for model_name, count in sorted(vote_counts.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {model_name:20s}: {count:4d} days")

        # Get best overall model
        best_model, best_params, best_score = self.model_selector.get_best_model()
        print(f"\nBest Overall Model:")
        print(f"  Model: {best_model}")
        print(f"  Score (MSE): {best_score:.6f}")
        print(f"  Parameters: {best_params}")

        # Get summary statistics
        summary_stats = self.model_selector.get_summary_statistics()
        print("\nPer-Model Statistics:")
        for model_name, stats in summary_stats.items():
            print(f"\n  {model_name}:")
            print(f"    Mean MSE:   {stats['mean_score']:.6f}")
            print(f"    Std MSE:    {stats['std_score']:.6f}")
            print(f"    Min MSE:    {stats['min_score']:.6f}")
            print(f"    Max MSE:    {stats['max_score']:.6f}")
            print(f"    Median MSE: {stats['median_score']:.6f}")
            print(f"    N Trials:   {stats['n_trials']}")

        return {
            'vote_counts': vote_counts,
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'summary_stats': summary_stats
        }


if __name__ == "__main__":
    print("Walk-Forward Tester Module")
    print("This module should be run via the main training script")
