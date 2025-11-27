"""
OHLC Tokenizer: Convert continuous price data to discrete tokens.
Uses adaptive quantization with k-means clustering on log returns.
"""

import numpy as np
import torch
import pickle
from typing import Optional, Tuple, Dict
from sklearn.cluster import KMeans
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TokenizerConfig:
    """Configuration for OHLC tokenizer."""
    vocab_size: int = 1024          # Number of discrete bins per feature
    n_features: int = 4              # OHLC features
    min_return: float = -0.1         # Minimum log return (cap outliers)
    max_return: float = 0.1          # Maximum log return (cap outliers)
    epsilon: float = 1e-8            # Small constant for numerical stability


class OHLCTokenizer:
    """
    Tokenizer for OHLC candlestick data.

    Converts continuous price values to discrete tokens using:
    1. Normalization: Convert to log returns
    2. Quantization: K-means clustering to create discrete bins
    3. Tokenization: Map values to token IDs

    This allows treating time series as sequences of discrete tokens,
    similar to text in NLP.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.is_fitted = False

        # K-means clusterers for each OHLC feature
        self.clusterers = [
            KMeans(n_clusters=config.vocab_size, random_state=42, n_init=10)
            for _ in range(config.n_features)
        ]

        # Store cluster centers for reconstruction
        self.cluster_centers = None

    def fit(self, ohlc_data: np.ndarray) -> 'OHLCTokenizer':
        """
        Fit the tokenizer on OHLC data.

        Args:
            ohlc_data: OHLC price data of shape (n_samples, 4)

        Returns:
            self
        """
        print("Fitting OHLC tokenizer...")
        print(f"  Data shape: {ohlc_data.shape}")
        print(f"  Vocab size: {self.config.vocab_size}")

        # Convert to log returns
        returns = self._compute_log_returns(ohlc_data)

        # Fit k-means for each feature
        self.cluster_centers = []

        for i, feature_name in enumerate(['Open', 'High', 'Low', 'Close']):
            feature_returns = returns[:, i:i+1]  # Keep 2D shape for sklearn

            # Fit k-means
            print(f"  Clustering {feature_name} returns...")
            self.clusterers[i].fit(feature_returns)

            # Store cluster centers
            centers = self.clusterers[i].cluster_centers_.flatten()
            self.cluster_centers.append(centers)

            # Print statistics
            print(f"    Return range: [{feature_returns.min():.6f}, {feature_returns.max():.6f}]")
            print(f"    Cluster centers range: [{centers.min():.6f}, {centers.max():.6f}]")

        self.is_fitted = True
        print("Tokenizer fitted successfully!\n")

        return self

    def encode(
        self,
        ohlc_data: np.ndarray,
        return_torch: bool = False
    ) -> np.ndarray:
        """
        Encode OHLC data to token IDs.

        Args:
            ohlc_data: OHLC price data of shape (n_samples, 4) or (batch, n_samples, 4)
            return_torch: If True, return torch.Tensor instead of numpy array

        Returns:
            Token IDs of same shape as input, with values in [0, vocab_size)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding. Call fit() first.")

        # Handle batched input
        original_shape = ohlc_data.shape
        if len(original_shape) == 3:
            batch_size, seq_len, n_feat = original_shape
            ohlc_data = ohlc_data.reshape(-1, n_feat)
        elif len(original_shape) != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {original_shape}")

        # Convert to log returns
        returns = self._compute_log_returns(ohlc_data)

        # Tokenize each feature
        token_ids = np.zeros((returns.shape[0], self.config.n_features), dtype=np.int64)

        for i in range(self.config.n_features):
            feature_returns = returns[:, i:i+1]
            token_ids[:, i] = self.clusterers[i].predict(feature_returns)

        # Reshape back to original batch shape if needed
        if len(original_shape) == 3:
            token_ids = token_ids.reshape(batch_size, seq_len, self.config.n_features)

        if return_torch:
            return torch.from_numpy(token_ids).long()

        return token_ids

    def decode(
        self,
        token_ids: np.ndarray,
        reference_prices: np.ndarray
    ) -> np.ndarray:
        """
        Decode token IDs back to OHLC prices.

        Args:
            token_ids: Token IDs of shape (n_samples, 4) or (batch, n_samples, 4)
            reference_prices: Reference prices to apply returns to, shape (n_samples, 4) or (batch, n_samples, 4)
                             First row is used as starting point for reconstruction

        Returns:
            Reconstructed OHLC prices
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding.")

        # Handle batched input
        original_shape = token_ids.shape
        if len(original_shape) == 3:
            batch_size, seq_len, n_feat = original_shape
            token_ids = token_ids.reshape(-1, n_feat)
            reference_prices = reference_prices.reshape(-1, n_feat)
        elif len(original_shape) != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {original_shape}")

        # Get returns from cluster centers
        returns = np.zeros_like(token_ids, dtype=np.float32)

        for i in range(self.config.n_features):
            returns[:, i] = self.cluster_centers[i][token_ids[:, i]]

        # Convert returns back to prices
        prices = self._returns_to_prices(returns, reference_prices)

        # Reshape back to original batch shape if needed
        if len(original_shape) == 3:
            prices = prices.reshape(original_shape)

        return prices

    def _compute_log_returns(self, ohlc_data: np.ndarray) -> np.ndarray:
        """
        Compute log returns from OHLC prices.

        Uses previous close as reference for each OHLC value.

        Args:
            ohlc_data: OHLC prices (n_samples, 4)

        Returns:
            Log returns (n_samples, 4)
        """
        n_samples = ohlc_data.shape[0]
        returns = np.zeros_like(ohlc_data, dtype=np.float32)

        # Get reference prices (previous close)
        ref_prices = np.roll(ohlc_data[:, 3], 1)  # Previous close
        ref_prices[0] = ohlc_data[0, 3]  # First candle: use own close as reference

        # Compute log returns for each feature
        for i in range(self.config.n_features):
            returns[:, i] = np.log((ohlc_data[:, i] + self.config.epsilon) /
                                  (ref_prices + self.config.epsilon))

        # Clip outliers
        returns = np.clip(returns, self.config.min_return, self.config.max_return)

        return returns

    def _returns_to_prices(
        self,
        returns: np.ndarray,
        reference_prices: np.ndarray
    ) -> np.ndarray:
        """
        Convert log returns back to prices.

        Args:
            returns: Log returns (n_samples, 4)
            reference_prices: Reference prices (n_samples, 4)

        Returns:
            Reconstructed prices (n_samples, 4)
        """
        prices = np.zeros_like(returns, dtype=np.float32)

        # Use previous close as reference
        ref_prices = np.roll(reference_prices[:, 3], 1)
        ref_prices[0] = reference_prices[0, 3]

        # Convert returns to prices
        for i in range(self.config.n_features):
            prices[:, i] = ref_prices * np.exp(returns[:, i])

        return prices

    def save(self, filepath: str):
        """Save tokenizer to disk."""
        state = {
            'config': self.config,
            'clusterers': self.clusterers,
            'cluster_centers': self.cluster_centers,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'OHLCTokenizer':
        """Load tokenizer from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        tokenizer = cls(state['config'])
        tokenizer.clusterers = state['clusterers']
        tokenizer.cluster_centers = state['cluster_centers']
        tokenizer.is_fitted = state['is_fitted']

        print(f"Tokenizer loaded from {filepath}")
        return tokenizer

    def get_vocab_stats(self) -> Dict:
        """Get statistics about the vocabulary."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted first.")

        stats = {
            'vocab_size': self.config.vocab_size,
            'n_features': self.config.n_features,
            'total_tokens': self.config.vocab_size * self.config.n_features,
        }

        for i, name in enumerate(['Open', 'High', 'Low', 'Close']):
            centers = self.cluster_centers[i]
            stats[f'{name}_center_range'] = (float(centers.min()), float(centers.max()))
            stats[f'{name}_center_mean'] = float(centers.mean())
            stats[f'{name}_center_std'] = float(centers.std())

        return stats


def create_sequences(
    token_ids: np.ndarray,
    sequence_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sequences for training.

    Args:
        token_ids: Token IDs of shape (n_samples, 4)
        sequence_length: Length of each sequence
        stride: Stride between sequences

    Returns:
        - X: Input sequences (n_sequences, sequence_length, 4)
        - y: Target tokens (n_sequences, 4) - next token after each sequence
    """
    n_samples = token_ids.shape[0]
    n_sequences = (n_samples - sequence_length) // stride

    X = np.zeros((n_sequences, sequence_length, token_ids.shape[1]), dtype=np.int64)
    y = np.zeros((n_sequences, token_ids.shape[1]), dtype=np.int64)

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        X[i] = token_ids[start_idx:end_idx]
        y[i] = token_ids[end_idx]  # Next token

    return X, y


if __name__ == "__main__":
    # Test tokenizer
    print("="*60)
    print("Testing OHLC Tokenizer")
    print("="*60)

    # Generate synthetic OHLC data
    n_samples = 10000
    base_price = 50000.0  # Bitcoin-like price

    # Random walk with some structure
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.002  # 0.2% std per candle
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    noise = np.random.randn(n_samples, 3) * 0.001
    ohlc_data = np.zeros((n_samples, 4))
    ohlc_data[:, 3] = close_prices  # Close
    ohlc_data[:, 0] = close_prices * (1 + noise[:, 0])  # Open
    ohlc_data[:, 1] = close_prices * (1 + abs(noise[:, 1]))  # High
    ohlc_data[:, 2] = close_prices * (1 - abs(noise[:, 2]))  # Low

    print(f"\nSynthetic OHLC data:")
    print(f"  Shape: {ohlc_data.shape}")
    print(f"  Price range: [{ohlc_data.min():.2f}, {ohlc_data.max():.2f}]")
    print(f"  Mean price: {ohlc_data.mean():.2f}")

    # Create and fit tokenizer
    config = TokenizerConfig(vocab_size=1024)
    tokenizer = OHLCTokenizer(config)
    tokenizer.fit(ohlc_data)

    # Test encoding
    print("\nTesting encoding...")
    token_ids = tokenizer.encode(ohlc_data[:100])
    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  Token ID range: [{token_ids.min()}, {token_ids.max()}]")
    print(f"  Sample tokens (first 5 candles):")
    for i in range(5):
        print(f"    Candle {i}: O={token_ids[i,0]:3d} H={token_ids[i,1]:3d} "
              f"L={token_ids[i,2]:3d} C={token_ids[i,3]:3d}")

    # Test decoding
    print("\nTesting decoding...")
    reconstructed = tokenizer.decode(token_ids, ohlc_data[:100])
    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Compute reconstruction error
    original = ohlc_data[:100]
    abs_error = np.abs(reconstructed - original)
    rel_error = abs_error / original * 100

    print(f"  Absolute error: {abs_error.mean():.4f} ± {abs_error.std():.4f}")
    print(f"  Relative error: {rel_error.mean():.4f}% ± {rel_error.std():.4f}%")

    # Test sequence creation
    print("\nTesting sequence creation...")
    X, y = create_sequences(token_ids, sequence_length=64, stride=1)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Test save/load
    print("\nTesting save/load...")
    tokenizer.save("/tmp/test_tokenizer.pkl")
    loaded_tokenizer = OHLCTokenizer.load("/tmp/test_tokenizer.pkl")

    # Verify loaded tokenizer works
    token_ids_loaded = loaded_tokenizer.encode(ohlc_data[:10])
    assert np.array_equal(token_ids[:10], token_ids_loaded), "Loaded tokenizer mismatch!"
    print("  Save/load test passed!")

    # Print vocabulary statistics
    print("\nVocabulary statistics:")
    stats = tokenizer.get_vocab_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("Tokenizer test completed successfully!")
    print("="*60)
