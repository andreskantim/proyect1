"""
Simplified tokenizer for price direction prediction.
Instead of predicting exact OHLC values, predict only direction (up/down/flat).
"""
import numpy as np
from typing import Tuple


class DirectionTokenizer:
    """
    Tokenizes price data into direction classes: DOWN (0), FLAT (1), UP (2)

    Much simpler than OHLC tokenization:
    - Only 3 classes instead of vocab_size^4
    - Only predicts close price direction
    - Easier to learn, more practical for trading
    """

    def __init__(self, threshold: float = 0.005):
        """
        Args:
            threshold: Minimum price change to consider as UP or DOWN.
                      Changes smaller than this are FLAT.
                      Default 0.005 = 0.5%
        """
        self.threshold = threshold
        self.classes = {0: "DOWN", 1: "FLAT", 2: "UP"}
        self.vocab_size = 3

    def fit(self, ohlc_data: np.ndarray):
        """
        No fitting needed for direction tokenizer.
        Just for compatibility with OHLC tokenizer interface.

        Args:
            ohlc_data: OHLC array (n_samples, 4)
        """
        print(f"DirectionTokenizer initialized with threshold={self.threshold}")
        print(f"  UP:   close_change > +{self.threshold*100:.2f}%")
        print(f"  FLAT: -{self.threshold*100:.2f}% <= close_change <= +{self.threshold*100:.2f}%")
        print(f"  DOWN: close_change < -{self.threshold*100:.2f}%")
        return self

    def transform(self, ohlc_data: np.ndarray) -> np.ndarray:
        """
        Convert OHLC data to direction tokens.

        Args:
            ohlc_data: OHLC array (n_samples, 4) - [open, high, low, close]

        Returns:
            direction_tokens: (n_samples, 1) with values {0, 1, 2}
        """
        close_prices = ohlc_data[:, 3]  # Extract close prices

        # Calculate returns
        returns = np.zeros(len(close_prices))
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

        # Tokenize: DOWN=0, FLAT=1, UP=2
        tokens = np.ones(len(returns), dtype=np.int64)  # Default: FLAT
        tokens[returns > self.threshold] = 2   # UP
        tokens[returns < -self.threshold] = 0  # DOWN

        # Reshape to (n_samples, 1)
        tokens = tokens.reshape(-1, 1)

        return tokens

    def inverse_transform(self, direction_tokens: np.ndarray) -> np.ndarray:
        """
        Convert direction tokens back to approximate returns.

        Args:
            direction_tokens: (n_samples, 1) with values {0, 1, 2}

        Returns:
            Approximate returns for each token
        """
        # DOWN -> -threshold, FLAT -> 0, UP -> +threshold
        returns = np.zeros(len(direction_tokens))

        flat_tokens = direction_tokens.flatten()
        returns[flat_tokens == 0] = -self.threshold
        returns[flat_tokens == 1] = 0.0
        returns[flat_tokens == 2] = self.threshold

        return returns

    def get_stats(self, direction_tokens: np.ndarray) -> dict:
        """Get distribution statistics of direction tokens."""
        flat_tokens = direction_tokens.flatten()

        total = len(flat_tokens)
        n_down = np.sum(flat_tokens == 0)
        n_flat = np.sum(flat_tokens == 1)
        n_up = np.sum(flat_tokens == 2)

        return {
            'total': total,
            'DOWN': n_down,
            'DOWN_pct': 100 * n_down / total,
            'FLAT': n_flat,
            'FLAT_pct': 100 * n_flat / total,
            'UP': n_up,
            'UP_pct': 100 * n_up / total,
        }


def create_direction_sequences(
    direction_tokens: np.ndarray,
    sequence_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for direction prediction.

    Args:
        direction_tokens: (n_samples, 1) direction tokens
        sequence_length: Length of input sequence
        stride: Stride between sequences

    Returns:
        X: (n_sequences, sequence_length, 1) - input sequences
        y: (n_sequences,) - target directions (next direction after sequence)
    """
    n_samples = direction_tokens.shape[0]
    n_sequences = (n_samples - sequence_length) // stride

    X = np.zeros((n_sequences, sequence_length, 1), dtype=np.int64)
    y = np.zeros(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length

        X[i] = direction_tokens[start_idx:end_idx]
        y[i] = direction_tokens[end_idx, 0]  # Next direction

    return X, y


if __name__ == "__main__":
    # Test direction tokenizer
    print("="*60)
    print("Testing Direction Tokenizer")
    print("="*60)

    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 1000

    # Simulate price walk
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.cumprod(1 + returns)

    # Create OHLC (simplified: all same as close)
    ohlc = np.column_stack([prices, prices, prices, prices])

    # Test tokenizer
    tokenizer = DirectionTokenizer(threshold=0.01)
    tokenizer.fit(ohlc)

    tokens = tokenizer.transform(ohlc)

    stats = tokenizer.get_stats(tokens)
    print(f"\nToken distribution:")
    print(f"  DOWN: {stats['DOWN']:4d} ({stats['DOWN_pct']:5.1f}%)")
    print(f"  FLAT: {stats['FLAT']:4d} ({stats['FLAT_pct']:5.1f}%)")
    print(f"  UP:   {stats['UP']:4d} ({stats['UP_pct']:5.1f}%)")

    # Test sequence creation
    X, y = create_direction_sequences(tokens, sequence_length=50, stride=1)
    print(f"\nSequences created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y distribution:")
    for i in range(3):
        count = np.sum(y == i)
        print(f"    {tokenizer.classes[i]}: {count} ({100*count/len(y):.1f}%)")
