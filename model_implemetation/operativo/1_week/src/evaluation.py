"""
Evaluation Metrics for Bitcoin Price Prediction

Implements performance measures from Masters (2018), Chapter 1:
- MSE, MAE, RMSE, R-squared
- Success ratios and profit factors (for trading)
- Spearman correlation (nonparametric)
- Confidence intervals using empirical quantiles

References:
    Masters, T. (2018). "Assessing and Improving Prediction and Classification"
    Chapter 1: Assessment of Numeric Predictions, pages 21-46
"""

import numpy as np
from scipy import stats
from scipy.special import btdtri
from typing import Dict, Tuple, Optional
import warnings


class PerformanceMetrics:
    """
    Comprehensive performance evaluation for numeric predictions.

    Implements metrics suitable for both general prediction and trading applications.
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (MSE).

        Reference: Masters (2018), p. 22

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE).

        More interpretable than MSE as it's in the same units as the data.

        Reference: Masters (2018), p. 25

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE value
        """
        return np.sqrt(PerformanceMetrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE).

        Less sensitive to outliers than MSE.

        Reference: Masters (2018), p. 24

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R-squared (coefficient of determination).

        Normalized measure: 0 = naive model, 1 = perfect prediction.

        Reference: Masters (2018), p. 24-25

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            R-squared value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Spearman rank correlation (nonparametric).

        Robust measure that only considers relative ordering.
        Excellent for cases where absolute accuracy is difficult.

        Reference: Masters (2018), p. 26-27

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Spearman correlation coefficient
        """
        correlation, _ = stats.spearmanr(y_true, y_pred)
        return correlation

    @staticmethod
    def success_ratio(returns: np.ndarray) -> float:
        """
        Success ratio for trading.

        Ratio of sum of profits to sum of losses.

        Reference: Masters (2018), p. 27-28, Equation 1.7

        Args:
            returns: Array of returns (positive = profit, negative = loss)

        Returns:
            Success ratio (0 to 1, where 1 = perfect)
        """
        profits = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0:
            return 1.0  # No losses = perfect

        if len(profits) == 0:
            return 0.0  # No profits = worst

        sum_profits = np.sum(profits)
        sum_losses = np.abs(np.sum(losses))

        # Formula: sum(profits) / (sum(profits) + sum(losses))
        return sum_profits / (sum_profits + sum_losses)

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Profit factor for trading.

        Ratio of gross profit to gross loss.

        Reference: Masters (2018), p. 27-28, Equation 1.6

        Args:
            returns: Array of returns (positive = profit, negative = loss)

        Returns:
            Profit factor (>1 is profitable, <1 is unprofitable)
        """
        profits = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0:
            return np.inf  # No losses

        if len(profits) == 0:
            return 0.0  # No profits

        sum_profits = np.sum(profits)
        sum_losses = np.abs(np.sum(losses))

        # Formula: sum(profits) / sum(losses)
        return sum_profits / sum_losses

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of correct directional predictions.

        For time series: did we predict the correct direction of change?

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Fraction of correct directions (0 to 1)
        """
        # Calculate changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Count correct directions
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)

        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all available metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            returns: Optional returns for trading metrics

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'mse': PerformanceMetrics.mse(y_true, y_pred),
            'rmse': PerformanceMetrics.rmse(y_true, y_pred),
            'mae': PerformanceMetrics.mae(y_true, y_pred),
            'r_squared': PerformanceMetrics.r_squared(y_true, y_pred),
            'spearman': PerformanceMetrics.spearman_correlation(y_true, y_pred),
            'directional_accuracy': PerformanceMetrics.directional_accuracy(y_true, y_pred)
        }

        if returns is not None:
            metrics['success_ratio'] = PerformanceMetrics.success_ratio(returns)
            metrics['profit_factor'] = PerformanceMetrics.profit_factor(returns)

        return metrics


class ConfidenceIntervals:
    """
    Confidence interval computation using empirical quantiles.

    Implements distribution-free confidence intervals as described in
    Masters (2018), Chapter 1, pages 38-46.
    """

    @staticmethod
    def empirical_quantile_ci(errors: np.ndarray, p: float = 0.05,
                             symmetric: bool = True) -> Tuple[float, float]:
        """
        Compute confidence interval using empirical quantiles.

        Reference: Masters (2018), p. 38-40

        Args:
            errors: Array of prediction errors
            p: Tail probability (default: 0.05 for 90% CI)
            symmetric: If True, split probability symmetrically

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        errors_sorted = np.sort(errors)
        n = len(errors)

        if symmetric:
            # Split probability equally in both tails
            m_lower = int(n * p)
            m_upper = int(n * (1 - p))
        else:
            # All probability in one tail
            m_lower = 0
            m_upper = int(n * (1 - p))

        # Conservative approach: use floor for indices
        m_lower = max(0, m_lower)
        m_upper = min(n - 1, m_upper)

        lower_bound = errors_sorted[m_lower]
        upper_bound = errors_sorted[m_upper]

        return lower_bound, upper_bound

    @staticmethod
    def quantile_confidence(n: int, m: int, confidence: float = 0.99) -> float:
        """
        Compute pessimistic quantile for given confidence level.

        Given n samples and using m-th order statistic, find q such that
        there's only (1-confidence) probability that true quantile exceeds q.

        Reference: Masters (2018), p. 41-43

        Args:
            n: Number of samples in confidence set
            m: Order statistic being used (e.g., 5th smallest)
            confidence: Desired confidence level (default: 0.99)

        Returns:
            Pessimistic quantile q
        """
        # This uses the incomplete beta distribution
        # We solve: 1 - I_q(m, n-m+1) = 1 - confidence
        # Therefore: I_q(m, n-m+1) = confidence

        # Use inverse incomplete beta to find q
        q = btdtri(m, n - m + 1, confidence)

        return q

    @staticmethod
    def tolerance_interval(n: int, m: int, gamma: float = 0.90) -> float:
        """
        Compute tolerance interval confidence.

        Find probability β that interval covers at least fraction γ of distribution.

        Reference: Masters (2018), p. 43-44, Equation 1.11

        Args:
            n: Number of samples
            m: Order statistics used for bounds (symmetric)
            gamma: Desired coverage (default: 0.90)

        Returns:
            Confidence β that coverage ≥ γ
        """
        # β = 1 - I_γ(n - 2m + 1, 2m)
        from scipy.special import betainc

        beta = 1 - betainc(n - 2*m + 1, 2*m, gamma)

        return beta

    @staticmethod
    def compute_confidence_bounds(errors: np.ndarray,
                                 confidence_level: float = 0.90,
                                 pessimistic_confidence: float = 0.99
                                 ) -> Dict[str, float]:
        """
        Compute comprehensive confidence bounds.

        Args:
            errors: Array of prediction errors from confidence set
            confidence_level: Desired CI coverage (default: 0.90)
            pessimistic_confidence: Confidence in pessimistic bound (default: 0.99)

        Returns:
            Dictionary with bounds and statistics
        """
        n = len(errors)
        p = (1 - confidence_level) / 2  # Tail probability

        # Compute empirical quantile bounds
        lower, upper = ConfidenceIntervals.empirical_quantile_ci(
            errors, p, symmetric=True
        )

        # Compute order statistic
        m = int(n * p)
        m = max(1, m)  # At least 1

        # Compute pessimistic quantile
        pessimistic_q = ConfidenceIntervals.quantile_confidence(
            n, m, pessimistic_confidence
        )

        # Compute tolerance interval confidence
        tolerance_conf = ConfidenceIntervals.tolerance_interval(
            n, m, confidence_level
        )

        return {
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_level': confidence_level,
            'n_samples': n,
            'order_statistic': m,
            'pessimistic_quantile': pessimistic_q,
            'tolerance_confidence': tolerance_conf
        }


def stratified_performance(errors: np.ndarray, n_strata: int = 10) -> Dict:
    """
    Compute performance consistency across strata.

    Stratification encourages consistent performance, which improves
    generalization (Masters 2018, p. 29-31).

    Args:
        errors: Array of errors
        n_strata: Number of strata to divide data into

    Returns:
        Dictionary with stratified statistics
    """
    n = len(errors)
    stratum_size = n // n_strata

    stratum_mses = []

    for i in range(n_strata):
        start = i * stratum_size
        end = (i + 1) * stratum_size if i < n_strata - 1 else n

        stratum_errors = errors[start:end]
        stratum_mse = np.mean(stratum_errors ** 2)
        stratum_mses.append(stratum_mse)

    stratum_mses = np.array(stratum_mses)

    # Consistency measure: mean/std ratio (higher is better)
    mean_mse = np.mean(stratum_mses)
    std_mse = np.std(stratum_mses)

    consistency_ratio = mean_mse / std_mse if std_mse > 0 else np.inf

    return {
        'n_strata': n_strata,
        'stratum_mses': stratum_mses,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'consistency_ratio': consistency_ratio,
        'worst_stratum_mse': np.max(stratum_mses),
        'best_stratum_mse': np.min(stratum_mses)
    }


if __name__ == "__main__":
    # Example usage
    print("Performance Metrics - Example Usage")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randn(n_samples).cumsum()
    y_pred = y_true + np.random.randn(n_samples) * 0.5  # Add some noise

    # Compute all metrics
    metrics = PerformanceMetrics.compute_all_metrics(y_true, y_pred)

    print("\nBasic Metrics:")
    for name, value in metrics.items():
        print(f"  {name:25s}: {value:.4f}")

    # Compute confidence intervals
    errors = y_pred - y_true
    ci_info = ConfidenceIntervals.compute_confidence_bounds(errors, 0.90, 0.99)

    print("\nConfidence Intervals (90%):")
    print(f"  Lower bound: {ci_info['lower_bound']:.4f}")
    print(f"  Upper bound: {ci_info['upper_bound']:.4f}")
    print(f"  Pessimistic quantile (99% conf): {ci_info['pessimistic_quantile']:.4f}")
    print(f"  Tolerance confidence: {ci_info['tolerance_confidence']:.4f}")

    # Stratified performance
    strat_perf = stratified_performance(errors, n_strata=10)
    print("\nStratified Performance:")
    print(f"  Consistency ratio: {strat_perf['consistency_ratio']:.4f}")
    print(f"  Worst stratum MSE: {strat_perf['worst_stratum_mse']:.4f}")
    print(f"  Best stratum MSE: {strat_perf['best_stratum_mse']:.4f}")
