"""
Feature Engineering for Bitcoin Price Prediction

This module creates technical indicators and features from OHLCV data.
Includes normalization and scaling appropriate for time-series walk-forward testing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional
import warnings


class FeatureEngineer:
    """
    Creates features from raw OHLCV candlestick data.

    Features include:
    - Raw OHLCV data
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price changes and returns
    - Volume indicators
    - Temporal features
    """

    def __init__(self, scaler_type='robust'):
        """
        Initialize feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'robust', or None)
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all features
        """
        features_df = df.copy()

        # 1. Price-based features
        features_df = self._add_price_features(features_df)

        # 2. Technical indicators
        features_df = self._add_technical_indicators(features_df)

        # 3. Volume features
        features_df = self._add_volume_features(features_df)

        # 4. Temporal features
        features_df = self._add_temporal_features(features_df)

        # Store feature names (excluding timestamp if present)
        self.feature_names = [col for col in features_df.columns
                              if col not in ['timestamp']]

        # Handle any NaN values from indicator calculations
        features_df = features_df.fillna(method='bfill').fillna(method='ffill')

        return features_df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()

        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']

        # Open-Close difference
        df['oc_diff'] = df['close'] - df['open']
        df['oc_diff_pct'] = (df['close'] - df['open']) / df['open']

        # Returns (log returns are better for multiplicative processes)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        # RSI (Relative Strength Index)
        df = self._calculate_rsi(df, period=14)
        df = self._calculate_rsi(df, period=7)

        # MACD (Moving Average Convergence Divergence)
        df = self._calculate_macd(df)

        # Bollinger Bands
        df = self._calculate_bollinger_bands(df, period=20)

        # Moving averages
        for period in [7, 14, 30, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Price position relative to moving averages
        df['price_to_sma_7'] = df['close'] / df['sma_7']
        df['price_to_sma_30'] = df['close'] / df['sma_30']

        # ATR (Average True Range)
        df = self._calculate_atr(df, period=14)

        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df[f'rsi_{period}'] = rsi

        return df

    def _calculate_macd(self, df: pd.DataFrame,
                       fast=12, slow=26, signal=9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame,
                                   period: int = 20,
                                   std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)

        # Bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume changes
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change()

        # Volume moving averages
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_sma_30'] = df['volume'].rolling(window=30).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_7']

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv

        # Price-Volume Trend
        df['pvt'] = ((df['close'].diff() / df['close'].shift()) * df['volume']).cumsum()

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/cyclical features."""
        # Hour of day (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # Day of week (cyclical encoding)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Day of month (cyclical encoding)
        df['dom_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df.index.day / 31)

        return df

    def fit_scaler(self, X: np.ndarray):
        """
        Fit scaler on training data.

        IMPORTANT: In walk-forward testing, this should be called only on
        training data to avoid future leak.

        Args:
            X: Training data of shape (n_samples, n_features)
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            # Robust scaler is better for data with outliers (Bitcoin!)
            self.scaler = RobustScaler()
        elif self.scaler_type is None:
            self.scaler = None
            return
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Reshape if needed (flatten time dimension for fitting)
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            self.scaler.fit(X_reshaped)
        else:
            self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data to transform

        Returns:
            Scaled data
        """
        if self.scaler is None:
            return X

        original_shape = X.shape

        # Reshape if 3D
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(original_shape)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data.

        Args:
            X: Data to fit and transform

        Returns:
            Scaled data
        """
        self.fit_scaler(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.

        Args:
            X: Scaled data

        Returns:
            Data in original scale
        """
        if self.scaler is None:
            return X

        original_shape = X.shape

        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_original = self.scaler.inverse_transform(X_reshaped)
            X_original = X_original.reshape(original_shape)
        else:
            X_original = self.scaler.inverse_transform(X)

        return X_original


def prepare_sequences_for_ml(X: np.ndarray, y: np.ndarray,
                             flatten_X: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for scikit-learn models.

    Most sklearn models expect 2D input, so we flatten the time dimension.

    Args:
        X: Input sequences (n_samples, n_timesteps, n_features)
        y: Target sequences (n_samples, n_timesteps, n_targets)
        flatten_X: If True, flatten X to (n_samples, n_timesteps * n_features)

    Returns:
        Tuple of (X_prepared, y_prepared)
    """
    if flatten_X and X.ndim == 3:
        # Flatten time and feature dimensions
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, n_timesteps * n_features)
    else:
        X_flat = X

    # For targets, we typically want to predict specific values
    # Here we'll predict the close price at each future timestep
    if y.ndim == 3:
        # Extract only close price (assuming it's the 4th column: OHLCV)
        y_close = y[:, :, 3]  # Shape: (n_samples, n_timesteps)
    else:
        y_close = y

    return X_flat, y_close


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering - Example Usage")
    print("=" * 50)

    # Create sample OHLCV data
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 10000 + np.random.randn(1000).cumsum() * 100,
        'high': 10100 + np.random.randn(1000).cumsum() * 100,
        'low': 9900 + np.random.randn(1000).cumsum() * 100,
        'close': 10000 + np.random.randn(1000).cumsum() * 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)

    # Create features
    engineer = FeatureEngineer(scaler_type='robust')
    features_df = engineer.create_features(df)

    print(f"\nOriginal columns: {list(df.columns)}")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"\nSample features:")
    print(features_df.head())
