#!/usr/bin/env python3
"""
Monte Carlo Simulation for Trading Strategy Validation

This script implements Monte Carlo simulation to assess the robustness
and risk of trading strategies based on model predictions.

Key Simulations:
1. Trade Resampling: Randomly resample trade sequences
2. Error Bootstrapping: Bootstrap prediction errors
3. Walk-Forward Monte Carlo: Multiple random walk-forward paths

Purpose:
- Estimate distribution of possible outcomes
- Calculate confidence intervals for performance metrics
- Assess worst-case and best-case scenarios
- Validate strategy robustness

Usage:
    python monte_carlo_test.py --predictions predictions.csv \
                                --actual actual.csv \
                                --n-simulations 10000 \
                                --output mc_results.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tqdm import tqdm
    import dask.bag as db
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install: pip install tqdm joblib")
    sys.exit(1)

from validation.utils import (
    calculate_sharpe_ratio,
    calculate_profit_factor,
    bootstrap_confidence_interval,
    print_validation_summary
)

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from dask_utils import get_dask_client, close_dask_client


class MonteCarloSimulation:
    """
    Monte Carlo Simulation for trading strategy validation.
    """

    def __init__(self,
                 n_simulations: int = 10000,
                 confidence_level: float = 0.95,
                 n_jobs: int = -1,
                 random_seed: int = 42,
                 verbose: bool = True):
        """
        Initialize Monte Carlo simulation.

        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for intervals (e.g., 0.95)
            n_jobs: Number of parallel jobs
            random_seed: Random seed
            verbose: Print progress
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose

        self.simulation_results = []

    def trade_resampling_simulation(self,
                                   returns: np.ndarray,
                                   simulation_idx: int) -> Dict[str, float]:
        """
        Resample trade sequence with replacement.

        This simulates different sequences of the same trades,
        testing if order matters.

        Args:
            returns: Array of trade returns
            simulation_idx: Simulation index for seeding

        Returns:
            Dictionary of performance metrics
        """
        seed = self.random_seed + simulation_idx
        np.random.seed(seed)

        # Resample returns with replacement
        resampled_returns = np.random.choice(returns, size=len(returns), replace=True)

        # Calculate metrics
        total_return = np.sum(resampled_returns)
        sharpe = calculate_sharpe_ratio(resampled_returns)
        max_drawdown = self._calculate_max_drawdown(resampled_returns)
        win_rate = np.mean(resampled_returns > 0)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def error_bootstrap_simulation(self,
                                   predictions: np.ndarray,
                                   actual: np.ndarray,
                                   simulation_idx: int) -> Dict[str, float]:
        """
        Bootstrap prediction errors to estimate uncertainty.

        Args:
            predictions: Model predictions
            actual: Actual values
            simulation_idx: Simulation index

        Returns:
            Dictionary of metrics
        """
        seed = self.random_seed + simulation_idx
        np.random.seed(seed)

        # Calculate errors
        errors = predictions - actual

        # Bootstrap errors
        bootstrap_errors = np.random.choice(errors, size=len(errors), replace=True)

        # Reconstruct predictions with bootstrapped errors
        bootstrap_predictions = actual + bootstrap_errors

        # Calculate trading returns
        returns = self._predictions_to_returns(bootstrap_predictions, actual)

        # Metrics
        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(actual, bootstrap_predictions)
        rmse = np.sqrt(mse)
        sharpe = calculate_sharpe_ratio(returns)
        total_return = np.sum(returns)

        return {
            'mse': mse,
            'rmse': rmse,
            'total_return': total_return,
            'sharpe_ratio': sharpe
        }

    def random_entry_simulation(self,
                               actual: np.ndarray,
                               simulation_idx: int,
                               hold_period: int = 24) -> Dict[str, float]:
        """
        Simulate random entry/exit points.

        This tests if any entry timing would work (baseline comparison).

        Args:
            actual: Actual price series
            simulation_idx: Simulation index
            hold_period: How long to hold each position

        Returns:
            Dictionary of metrics
        """
        seed = self.random_seed + simulation_idx
        np.random.seed(seed)

        returns = []
        i = 0

        while i < len(actual) - hold_period:
            # Random decision: long (1), short (-1), or neutral (0)
            decision = np.random.choice([1, -1, 0], p=[0.4, 0.4, 0.2])

            if decision != 0:
                # Calculate return over hold period
                entry_price = actual[i]
                exit_price = actual[i + hold_period]
                ret = decision * (exit_price - entry_price) / entry_price
                returns.append(ret)

            i += hold_period

        if len(returns) > 0:
            returns_array = np.array(returns)
            total_return = np.sum(returns_array)
            sharpe = calculate_sharpe_ratio(returns_array)
            win_rate = np.mean(returns_array > 0)
        else:
            total_return = 0
            sharpe = 0
            win_rate = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'n_trades': len(returns)
        }

    def _predictions_to_returns(self,
                               predictions: np.ndarray,
                               actual: np.ndarray) -> np.ndarray:
        """
        Convert predictions to trading returns.

        Strategy: Go long if predict price increase, flat otherwise.

        Args:
            predictions: Predicted prices
            actual: Actual prices

        Returns:
            Array of returns
        """
        returns = []

        for i in range(len(predictions) - 1):
            # Predict direction
            predicted_direction = predictions[i+1] - predictions[i]

            if predicted_direction > 0:  # Predict up -> go long
                ret = (actual[i+1] - actual[i]) / actual[i]
            else:  # Predict down or flat -> stay flat
                ret = 0

            returns.append(ret)

        return np.array(returns)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)

        return max_drawdown

    def run_trade_resampling(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Run trade resampling Monte Carlo simulation.

        Args:
            returns: Array of trade returns

        Returns:
            Dictionary with simulation results
        """
        if self.verbose:
            print(f"\nRunning {self.n_simulations} trade resampling simulations...")

        tasks = [(returns, i) for i in range(self.n_simulations)]
        bag = db.from_sequence(tasks, partition_size=100)
        results = bag.map(lambda x: self.trade_resampling_simulation(*x)).compute()

        # Aggregate results
        df_results = pd.DataFrame(results)

        summary = {
            'simulation_type': 'trade_resampling',
            'n_simulations': self.n_simulations,
            'metrics': {}
        }

        for col in df_results.columns:
            values = df_results[col].values
            summary['metrics'][col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_lower': float(np.percentile(values, (1 - self.confidence_level) / 2 * 100)),
                'ci_upper': float(np.percentile(values, (1 + self.confidence_level) / 2 * 100))
            }

        return summary

    def run_error_bootstrap(self,
                           predictions: np.ndarray,
                           actual: np.ndarray) -> Dict[str, Any]:
        """
        Run error bootstrapping Monte Carlo simulation.

        Args:
            predictions: Model predictions
            actual: Actual values

        Returns:
            Dictionary with simulation results
        """
        if self.verbose:
            print(f"\nRunning {self.n_simulations} error bootstrap simulations...")

        tasks = [(predictions, actual, i) for i in range(self.n_simulations)]
        bag = db.from_sequence(tasks, partition_size=100)
        results = bag.map(lambda x: self.error_bootstrap_simulation(*x)).compute()

        df_results = pd.DataFrame(results)

        summary = {
            'simulation_type': 'error_bootstrap',
            'n_simulations': self.n_simulations,
            'metrics': {}
        }

        for col in df_results.columns:
            values = df_results[col].values
            summary['metrics'][col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_lower': float(np.percentile(values, (1 - self.confidence_level) / 2 * 100)),
                'ci_upper': float(np.percentile(values, (1 + self.confidence_level) / 2 * 100))
            }

        return summary

    def run_random_entry(self, actual: np.ndarray, hold_period: int = 24) -> Dict[str, Any]:
        """
        Run random entry Monte Carlo simulation.

        Args:
            actual: Actual price series
            hold_period: Hold period for each trade

        Returns:
            Dictionary with simulation results
        """
        if self.verbose:
            print(f"\nRunning {self.n_simulations} random entry simulations...")

        tasks = [(actual, i, hold_period) for i in range(self.n_simulations)]
        bag = db.from_sequence(tasks, partition_size=100)
        results = bag.map(lambda x: self.random_entry_simulation(*x)).compute()

        df_results = pd.DataFrame(results)

        summary = {
            'simulation_type': 'random_entry',
            'n_simulations': self.n_simulations,
            'hold_period': hold_period,
            'metrics': {}
        }

        for col in df_results.columns:
            values = df_results[col].values
            summary['metrics'][col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_lower': float(np.percentile(values, (1 - self.confidence_level) / 2 * 100)),
                'ci_upper': float(np.percentile(values, (1 + self.confidence_level) / 2 * 100))
            }

        return summary


def main():
    client = get_dask_client()

    parser = argparse.ArgumentParser(
        description='Monte Carlo Simulation for Trading Strategy Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file (or column name in data file)'
    )

    parser.add_argument(
        '--actual',
        type=str,
        required=True,
        help='Path to actual values CSV file (or column name in data file)'
    )

    parser.add_argument(
        '--n-simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )

    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for intervals (default: 0.95)'
    )

    parser.add_argument(
        '--simulation-type',
        type=str,
        default='all',
        choices=['trade_resampling', 'error_bootstrap', 'random_entry', 'all'],
        help='Type of simulation to run (default: all)'
    )

    parser.add_argument(
        '--hold-period',
        type=int,
        default=24,
        help='Hold period for random entry simulation (default: 24 hours)'
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
        default='monte_carlo_results.json',
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    print("="*80)
    print("MONTE CARLO SIMULATION")
    print("="*80)
    print(f"Predictions: {args.predictions}")
    print(f"Actual: {args.actual}")
    print(f"Simulations: {args.n_simulations}")
    print(f"Confidence Level: {args.confidence_level}")
    print(f"Simulation Type: {args.simulation_type}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    predictions = pd.read_csv(args.predictions).values.flatten()
    actual = pd.read_csv(args.actual).values.flatten()

    print(f"Predictions shape: {predictions.shape}")
    print(f"Actual shape: {actual.shape}")

    # Initialize simulation
    mc = MonteCarloSimulation(
        n_simulations=args.n_simulations,
        confidence_level=args.confidence_level,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        verbose=True
    )

    # Run simulations
    results = {
        'simulation_info': {
            'n_simulations': args.n_simulations,
            'confidence_level': args.confidence_level,
            'random_seed': args.seed,
            'timestamp': datetime.now().isoformat()
        },
        'simulations': {}
    }

    if args.simulation_type in ['trade_resampling', 'all']:
        # Calculate returns from predictions
        returns = mc._predictions_to_returns(predictions, actual)
        results['simulations']['trade_resampling'] = mc.run_trade_resampling(returns)

    if args.simulation_type in ['error_bootstrap', 'all']:
        results['simulations']['error_bootstrap'] = mc.run_error_bootstrap(predictions, actual)

    if args.simulation_type in ['random_entry', 'all']:
        results['simulations']['random_entry'] = mc.run_random_entry(actual, args.hold_period)

    # Print results
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*80)

    for sim_type, sim_results in results['simulations'].items():
        print(f"\n{sim_type.replace('_', ' ').title()}:")
        print("-" * 80)

        for metric, stats in sim_results['metrics'].items():
            print(f"\n  {metric.replace('_', ' ').title()}:")
            print(f"    Mean:   {stats['mean']:.6f}")
            print(f"    Median: {stats['median']:.6f}")
            print(f"    Std:    {stats['std']:.6f}")
            print(f"    Min:    {stats['min']:.6f}")
            print(f"    Max:    {stats['max']:.6f}")
            print(f"    CI {args.confidence_level*100:.0f}%:  [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]")

    print("\n" + "="*80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    close_dask_client(client)


if __name__ == "__main__":
    main()
