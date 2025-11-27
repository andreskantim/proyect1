"""
Multi-task tokenizer for market prediction.

Predicts:
1. Direction of next candle (UP/FLAT/DOWN)
2. Magnitude: whether price exceeds 2σ Bollinger in multiple horizons
   - 1 week (5 trading days)
   - 2 weeks (10 trading days)
   - 1 month (20 trading days)
   - 2 months (40 trading days)
"""
import numpy as np
from typing import Tuple, Dict, List


class MultiTaskTokenizer:
    """
    Tokenizes price data for multi-task prediction:

    Task 1: Direction Classification (3 classes)
        - DOWN (0): next candle closes lower
        - FLAT (1): next candle closes similar (within threshold)
        - UP (2): next candle closes higher

    Task 2: Magnitude Classification (4 binary tasks)
        - For each horizon (1w, 2w, 1m, 2m), predict whether:
          - Price exceeds +2σ Bollinger (bullish signal)
          - Price falls below -2σ Bollinger (bearish signal)
        - Binary: 0 = no significant movement, 1 = exceeds 2σ threshold
    """

    def __init__(
        self,
        direction_threshold: float = 0.005,
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
        horizons: List[int] = None
    ):
        """
        Args:
            direction_threshold: Minimum % change for UP/DOWN (default 0.5%)
            bollinger_window: Window for Bollinger bands calculation (default 20)
            bollinger_std: Standard deviations for Bollinger bands (default 2.0)
            horizons: List of horizons in days [1w, 2w, 1m, 2m]
        """
        self.direction_threshold = direction_threshold
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std

        # Default horizons: 1w, 2w, 1m, 2m (in trading days)
        self.horizons = horizons or [5, 10, 20, 40]
        self.horizon_names = ['1w', '2w', '1m', '2m']

        # Direction classes
        self.direction_classes = {0: "DOWN", 1: "FLAT", 2: "UP"}
        self.n_direction_classes = 3

        # Magnitude classes (binary per horizon)
        self.n_magnitude_classes = 2  # [no_signal, exceeds_2sigma]
        self.n_horizons = len(self.horizons)

        # Bollinger bands (fitted on training data)
        self.bb_middle = None
        self.bb_upper = None
        self.bb_lower = None

    def fit(self, ohlc_data: np.ndarray):
        """
        Fit the tokenizer on training data.

        Args:
            ohlc_data: OHLC array (n_samples, 4) - [open, high, low, close]
        """
        close_prices = ohlc_data[:, 3]

        # Calculate Bollinger Bands on training data
        self.bb_middle, self.bb_upper, self.bb_lower = self._calculate_bollinger_bands(
            close_prices
        )

        print(f"\nMultiTaskTokenizer initialized:")
        print(f"  Direction prediction:")
        print(f"    UP:   close_change > +{self.direction_threshold*100:.2f}%")
        print(f"    FLAT: -{self.direction_threshold*100:.2f}% <= close_change <= +{self.direction_threshold*100:.2f}%")
        print(f"    DOWN: close_change < -{self.direction_threshold*100:.2f}%")
        print(f"\n  Magnitude prediction (2σ Bollinger):")
        print(f"    Bollinger window: {self.bollinger_window} periods")
        print(f"    Bollinger std: {self.bollinger_std}σ")
        print(f"    Horizons: {', '.join([f'{h}d ({n})' for h, n in zip(self.horizons, self.horizon_names)])}")

        return self

    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Array of close prices

        Returns:
            middle, upper, lower bands
        """
        middle = np.zeros(len(prices))
        std = np.zeros(len(prices))

        # Calculate rolling mean and std
        for i in range(len(prices)):
            if i < self.bollinger_window:
                # Use all available data for first window
                window = prices[:i+1]
            else:
                window = prices[i-self.bollinger_window+1:i+1]

            middle[i] = np.mean(window)
            std[i] = np.std(window)

        upper = middle + self.bollinger_std * std
        lower = middle - self.bollinger_std * std

        return middle, upper, lower

    def transform(self, ohlc_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform OHLC data into multi-task targets.

        Args:
            ohlc_data: OHLC array (n_samples, 4) - [open, high, low, close]

        Returns:
            Dictionary with:
                - 'direction': (n_samples,) with values {0, 1, 2}
                - 'magnitude': (n_samples, n_horizons) with binary values
        """
        close_prices = ohlc_data[:, 3]
        n_samples = len(close_prices)

        # Task 1: Direction of next candle
        direction = self._compute_direction(close_prices)

        # Task 2: Magnitude across horizons
        magnitude = self._compute_magnitude(close_prices)

        return {
            'direction': direction,
            'magnitude': magnitude
        }

    def _compute_direction(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Compute direction tokens for next candle.

        Args:
            close_prices: Array of close prices

        Returns:
            direction_tokens: (n_samples,) with values {0, 1, 2}
        """
        n_samples = len(close_prices)

        # Calculate returns
        returns = np.zeros(n_samples)
        returns[:-1] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

        # Tokenize: DOWN=0, FLAT=1, UP=2
        tokens = np.ones(n_samples, dtype=np.int64)  # Default: FLAT
        tokens[returns > self.direction_threshold] = 2   # UP
        tokens[returns < -self.direction_threshold] = 0  # DOWN

        # Last sample has no next candle - set to FLAT
        tokens[-1] = 1

        return tokens

    def _compute_magnitude(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Compute magnitude tokens for multiple horizons.

        For each horizon, check if future price exceeds 2σ Bollinger bands.

        Args:
            close_prices: Array of close prices

        Returns:
            magnitude_tokens: (n_samples, n_horizons) with binary values
        """
        n_samples = len(close_prices)
        magnitude = np.zeros((n_samples, self.n_horizons), dtype=np.int64)

        # Calculate Bollinger bands for this data
        bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(close_prices)

        for horizon_idx, horizon in enumerate(self.horizons):
            for i in range(n_samples):
                # Check if any future price (within horizon) exceeds 2σ
                end_idx = min(i + horizon, n_samples)

                if end_idx > i:
                    future_prices = close_prices[i:end_idx]

                    # Check if price exceeds upper or lower Bollinger band
                    exceeds_upper = np.any(future_prices > bb_upper[i])
                    exceeds_lower = np.any(future_prices < bb_lower[i])

                    if exceeds_upper or exceeds_lower:
                        magnitude[i, horizon_idx] = 1  # Significant movement
                    else:
                        magnitude[i, horizon_idx] = 0  # No significant movement
                else:
                    # Not enough future data
                    magnitude[i, horizon_idx] = 0

        return magnitude

    def get_stats(self, targets: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Get distribution statistics of targets.

        Args:
            targets: Dictionary with 'direction' and 'magnitude'

        Returns:
            Statistics dictionary
        """
        direction = targets['direction']
        magnitude = targets['magnitude']

        total = len(direction)

        # Direction statistics
        n_down = np.sum(direction == 0)
        n_flat = np.sum(direction == 1)
        n_up = np.sum(direction == 2)

        # Magnitude statistics (per horizon)
        magnitude_stats = {}
        for horizon_idx, horizon_name in enumerate(self.horizon_names):
            n_significant = np.sum(magnitude[:, horizon_idx] == 1)
            magnitude_stats[horizon_name] = {
                'significant': n_significant,
                'significant_pct': 100 * n_significant / total,
                'no_signal': total - n_significant,
                'no_signal_pct': 100 * (total - n_significant) / total
            }

        return {
            'total_samples': total,
            'direction': {
                'DOWN': n_down,
                'DOWN_pct': 100 * n_down / total,
                'FLAT': n_flat,
                'FLAT_pct': 100 * n_flat / total,
                'UP': n_up,
                'UP_pct': 100 * n_up / total,
            },
            'magnitude': magnitude_stats
        }


def create_multitask_sequences(
    ohlc_data: np.ndarray,
    targets: Dict[str, np.ndarray],
    sequence_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create sequences for multi-task prediction.

    Args:
        ohlc_data: (n_samples, 4) OHLC data
        targets: Dictionary with 'direction' and 'magnitude' targets
        sequence_length: Length of input sequence
        stride: Stride between sequences

    Returns:
        X: (n_sequences, sequence_length, 4) - input OHLC sequences
        y: Dictionary with target arrays:
            - 'direction': (n_sequences,) - next candle direction
            - 'magnitude': (n_sequences, n_horizons) - magnitude signals
    """
    n_samples = ohlc_data.shape[0]
    n_sequences = (n_samples - sequence_length) // stride
    n_horizons = targets['magnitude'].shape[1]

    X = np.zeros((n_sequences, sequence_length, 4), dtype=np.float32)
    y_direction = np.zeros(n_sequences, dtype=np.int64)
    y_magnitude = np.zeros((n_sequences, n_horizons), dtype=np.int64)

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length

        # Input: OHLC sequence
        X[i] = ohlc_data[start_idx:end_idx]

        # Target: prediction for the candle AFTER the sequence
        y_direction[i] = targets['direction'][end_idx]
        y_magnitude[i] = targets['magnitude'][end_idx]

    return X, {
        'direction': y_direction,
        'magnitude': y_magnitude
    }


if __name__ == "__main__":
    # Test multi-task tokenizer
    print("="*70)
    print("Testing Multi-Task Tokenizer")
    print("="*70)

    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 2000

    # Simulate price walk with trending behavior
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.cumprod(1 + returns)

    # Create OHLC (simplified: all same as close)
    ohlc = np.column_stack([prices, prices * 1.01, prices * 0.99, prices])

    # Test tokenizer
    tokenizer = MultiTaskTokenizer(
        direction_threshold=0.005,
        bollinger_window=20,
        bollinger_std=2.0
    )
    tokenizer.fit(ohlc)

    targets = tokenizer.transform(ohlc)

    stats = tokenizer.get_stats(targets)

    print(f"\n{'='*70}")
    print("DIRECTION PREDICTION STATISTICS")
    print('='*70)
    print(f"  DOWN: {stats['direction']['DOWN']:4d} ({stats['direction']['DOWN_pct']:5.1f}%)")
    print(f"  FLAT: {stats['direction']['FLAT']:4d} ({stats['direction']['FLAT_pct']:5.1f}%)")
    print(f"  UP:   {stats['direction']['UP']:4d} ({stats['direction']['UP_pct']:5.1f}%)")

    print(f"\n{'='*70}")
    print("MAGNITUDE PREDICTION STATISTICS (2σ Bollinger)")
    print('='*70)
    for horizon_name, horizon_stats in stats['magnitude'].items():
        print(f"\n  {horizon_name}:")
        print(f"    Significant movement: {horizon_stats['significant']:4d} ({horizon_stats['significant_pct']:5.1f}%)")
        print(f"    No signal:           {horizon_stats['no_signal']:4d} ({horizon_stats['no_signal_pct']:5.1f}%)")

    # Test sequence creation
    print(f"\n{'='*70}")
    print("SEQUENCE CREATION")
    print('='*70)
    X, y = create_multitask_sequences(ohlc, targets, sequence_length=50, stride=1)
    print(f"  X shape: {X.shape}")
    print(f"  y['direction'] shape: {y['direction'].shape}")
    print(f"  y['magnitude'] shape: {y['magnitude'].shape}")

    print(f"\n  Direction distribution in sequences:")
    for i in range(3):
        count = np.sum(y['direction'] == i)
        print(f"    {tokenizer.direction_classes[i]}: {count} ({100*count/len(y['direction']):.1f}%)")

    print(f"\n  Magnitude distribution in sequences:")
    for horizon_idx, horizon_name in enumerate(tokenizer.horizon_names):
        n_significant = np.sum(y['magnitude'][:, horizon_idx] == 1)
        print(f"    {horizon_name}: {n_significant} significant ({100*n_significant/len(y['magnitude']):.1f}%)")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print('='*70)
