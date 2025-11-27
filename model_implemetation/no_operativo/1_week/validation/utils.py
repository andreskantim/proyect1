"""
Validation Utilities

Common functions for Monte Carlo Permutation Tests and validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings


def permute_returns(prices: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Permute price data by shuffling returns while preserving statistical properties.

    This method:
    1. Calculates log returns from prices
    2. Randomly shuffles the returns
    3. Reconstructs prices from shuffled returns

    This preserves volatility and return distribution while destroying
    temporal patterns and autocorrelation.

    Args:
        prices: Array of prices (close prices)
        seed: Random seed for reproducibility

    Returns:
        Array of permuted prices
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate log returns
    log_returns = np.diff(np.log(prices))

    # Shuffle returns
    permuted_returns = np.random.permutation(log_returns)

    # Reconstruct prices from shuffled returns
    permuted_prices = prices[0] * np.exp(np.cumsum(np.concatenate([[0], permuted_returns])))

    return permuted_prices


def permute_ohlcv_bars(df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Permute OHLCV data by shuffling intrabar price changes.

    This sophisticated permutation method:
    1. Calculates returns for each OHLCV component
    2. Shuffles the returns
    3. Reconstructs OHLCV bars maintaining valid relationships (High >= Low, etc.)

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        seed: Random seed for reproducibility

    Returns:
        DataFrame with permuted OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    df_copy = df.copy()

    # Permute close prices (main price series)
    close_permuted = permute_returns(df['close'].values, seed=seed)

    # Calculate relative positions of OHLC within each bar
    close_prices = df['close'].values
    open_ratio = df['open'].values / close_prices
    high_ratio = df['high'].values / close_prices
    low_ratio = df['low'].values / close_prices

    # Shuffle ratios independently to break temporal patterns
    if seed is not None:
        np.random.seed(seed + 1)
    open_ratio_shuffled = np.random.permutation(open_ratio)

    if seed is not None:
        np.random.seed(seed + 2)
    high_ratio_shuffled = np.random.permutation(high_ratio)

    if seed is not None:
        np.random.seed(seed + 3)
    low_ratio_shuffled = np.random.permutation(low_ratio)

    # Reconstruct OHLC maintaining valid relationships
    df_copy['close'] = close_permuted
    df_copy['open'] = close_permuted * open_ratio_shuffled

    # Ensure high is actually the highest and low is the lowest
    high_candidate = close_permuted * high_ratio_shuffled
    low_candidate = close_permuted * low_ratio_shuffled

    df_copy['high'] = np.maximum.reduce([
        df_copy['open'],
        df_copy['close'],
        high_candidate
    ])

    df_copy['low'] = np.minimum.reduce([
        df_copy['open'],
        df_copy['close'],
        low_candidate
    ])

    # Permute volume independently
    if seed is not None:
        np.random.seed(seed + 4)
    df_copy['volume'] = np.random.permutation(df['volume'].values)

    return df_copy


def block_permutation(data: np.ndarray, block_size: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Perform block permutation to preserve short-term dependencies.

    Instead of shuffling individual observations, shuffle blocks of data.
    This preserves intra-block patterns while destroying longer-term patterns.

    Args:
        data: Array of data to permute
        block_size: Size of blocks to shuffle
        seed: Random seed for reproducibility

    Returns:
        Block-permuted data
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    n_blocks = n // block_size

    # Create blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        blocks.append(data[start:end])

    # Handle remainder
    remainder_start = n_blocks * block_size
    if remainder_start < n:
        blocks.append(data[remainder_start:])

    # Shuffle blocks
    np.random.shuffle(blocks)

    # Concatenate shuffled blocks
    permuted_data = np.concatenate(blocks)

    return permuted_data


def calculate_p_value(real_metric: float,
                     permuted_metrics: List[float],
                     alternative: str = 'greater') -> float:
    """
    Calculate p-value from permutation test results.

    Args:
        real_metric: The metric value from real data
        permuted_metrics: List of metric values from permutations
        alternative: 'greater' (real > permuted), 'less' (real < permuted),
                    or 'two-sided'

    Returns:
        p-value
    """
    permuted_array = np.array(permuted_metrics)
    n_permutations = len(permuted_array)

    if alternative == 'greater':
        # How many permutations achieved >= real metric?
        n_extreme = np.sum(permuted_array >= real_metric)
        p_value = (n_extreme + 1) / (n_permutations + 1)

    elif alternative == 'less':
        # How many permutations achieved <= real metric?
        n_extreme = np.sum(permuted_array <= real_metric)
        p_value = (n_extreme + 1) / (n_permutations + 1)

    elif alternative == 'two-sided':
        # How many permutations are as extreme as real metric?
        mean_permuted = np.mean(permuted_array)
        real_deviation = abs(real_metric - mean_permuted)
        permuted_deviations = np.abs(permuted_array - mean_permuted)
        n_extreme = np.sum(permuted_deviations >= real_deviation)
        p_value = (n_extreme + 1) / (n_permutations + 1)

    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return p_value


def calculate_percentile_rank(real_metric: float,
                              permuted_metrics: List[float]) -> float:
    """
    Calculate percentile rank of real metric within permuted distribution.

    Args:
        real_metric: The metric value from real data
        permuted_metrics: List of metric values from permutations

    Returns:
        Percentile rank (0-100)
    """
    permuted_array = np.array(permuted_metrics)
    n_worse = np.sum(permuted_array <= real_metric)
    percentile = (n_worse / len(permuted_array)) * 100

    return percentile


def bootstrap_confidence_interval(data: np.ndarray,
                                  statistic_func,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95,
                                  seed: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Input data
        statistic_func: Function to calculate statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    bootstrap_stats = []

    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_stat = np.mean(bootstrap_stats)
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return mean_stat, lower_bound, upper_bound


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    sharpe = (mean_return - risk_free_rate) / std_return

    return sharpe


def calculate_profit_factor(predictions: np.ndarray,
                           actual: np.ndarray,
                           threshold: float = 0.0) -> float:
    """
    Calculate profit factor for predictions.

    Profit Factor = Sum of Profits / |Sum of Losses|

    A profit/loss is determined by whether prediction correctly
    predicted direction of price movement.

    Args:
        predictions: Predicted values
        actual: Actual values
        threshold: Threshold for considering a prediction correct

    Returns:
        Profit factor (> 1 is good, < 1 is bad)
    """
    # Calculate prediction errors
    errors = predictions - actual

    # Calculate profits and losses based on prediction direction
    actual_changes = np.diff(actual)
    predicted_changes = np.diff(predictions)

    # If prediction and actual have same sign, it's a "profit"
    correct_direction = np.sign(predicted_changes) == np.sign(actual_changes)

    profits = np.abs(actual_changes[correct_direction])
    losses = np.abs(actual_changes[~correct_direction])

    total_profit = np.sum(profits) if len(profits) > 0 else 0
    total_loss = np.sum(losses) if len(losses) > 0 else 1e-10  # Avoid division by zero

    profit_factor = total_profit / total_loss

    return profit_factor


def validate_predictions(predictions: np.ndarray,
                        actual: np.ndarray) -> dict:
    """
    Validate predictions and calculate multiple metrics.

    Args:
        predictions: Predicted values
        actual: Actual values

    Returns:
        Dictionary of validation metrics
    """
    from scipy import stats
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {}

    # Basic metrics
    metrics['mse'] = mean_squared_error(actual, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(actual, predictions)
    metrics['r2'] = r2_score(actual, predictions)

    # Directional accuracy
    actual_direction = np.sign(np.diff(actual))
    pred_direction = np.sign(np.diff(predictions))
    metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)

    # Correlation
    metrics['pearson_correlation'] = stats.pearsonr(actual, predictions)[0]
    metrics['spearman_correlation'] = stats.spearmanr(actual, predictions)[0]

    # Trading metrics
    metrics['profit_factor'] = calculate_profit_factor(predictions, actual)

    # Calculate returns if we traded based on predictions
    returns = []
    for i in range(len(predictions) - 1):
        if predictions[i+1] > predictions[i]:  # Predict up
            ret = (actual[i+1] - actual[i]) / actual[i]
        else:  # Predict down or neutral
            ret = 0
        returns.append(ret)

    if len(returns) > 0:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(np.array(returns))
        metrics['total_return'] = np.sum(returns)
        metrics['mean_return'] = np.mean(returns)

    return metrics


def print_validation_summary(metrics: dict, title: str = "Validation Results"):
    """
    Print a formatted summary of validation metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

    print("\nPrediction Accuracy Metrics:")
    print(f"  MSE:                    {metrics.get('mse', 0):.6f}")
    print(f"  RMSE:                   {metrics.get('rmse', 0):.6f}")
    print(f"  MAE:                    {metrics.get('mae', 0):.6f}")
    print(f"  R²:                     {metrics.get('r2', 0):.6f}")

    print("\nDirectional Metrics:")
    print(f"  Directional Accuracy:   {metrics.get('directional_accuracy', 0):.2%}")
    print(f"  Pearson Correlation:    {metrics.get('pearson_correlation', 0):.6f}")
    print(f"  Spearman Correlation:   {metrics.get('spearman_correlation', 0):.6f}")

    print("\nTrading Performance Metrics:")
    print(f"  Profit Factor:          {metrics.get('profit_factor', 0):.4f}")
    print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Total Return:           {metrics.get('total_return', 0):.4%}")
    print(f"  Mean Return:            {metrics.get('mean_return', 0):.6%}")

    if 'p_value' in metrics:
        print("\nStatistical Significance:")
        print(f"  p-value:                {metrics['p_value']:.6f}")
        print(f"  Percentile Rank:        {metrics.get('percentile_rank', 0):.2f}%")

        if metrics['p_value'] < 0.01:
            sig_level = "*** (p < 0.01)"
        elif metrics['p_value'] < 0.05:
            sig_level = "** (p < 0.05)"
        elif metrics['p_value'] < 0.10:
            sig_level = "* (p < 0.10)"
        else:
            sig_level = "(not significant)"

        print(f"  Significance:           {sig_level}")

    print("="*80)
