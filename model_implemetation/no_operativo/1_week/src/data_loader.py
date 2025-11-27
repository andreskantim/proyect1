"""
Data Loader for Bitcoin Hourly Candles

This module handles loading and preprocessing Bitcoin hourly candlestick data.
Implements proper dataset splitting with overlap management as described in
Masters (2018), Chapter 1, pages 16-18.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Dict
import warnings


class BitcoinDataLoader:
    """
    Loads and prepares Bitcoin hourly candlestick data for walk-forward testing.

    Attributes:
        lookback_hours (int): Number of hours to use as input (168 = 7 days)
        lookahead_hours (int): Number of hours to predict (24 = 1 day)
        gap_days (int): Gap between datasets to prevent overlap (default: 1 day)
    """

    def __init__(self, lookback_hours=168, lookahead_hours=24, gap_days=1):
        """
        Initialize the data loader.

        Args:
            lookback_hours: Hours of historical data for features (default: 168 = 7 days)
            lookahead_hours: Hours to predict ahead (default: 24 = 1 day)
            gap_days: Gap in days between datasets to avoid overlap (default: 1)
                     Per Masters (2018), required gap = min(lookback, lookahead) - 1 = 23h
                     So 1 day (24h) is sufficient.
        """
        self.lookback_hours = lookback_hours
        self.lookahead_hours = lookahead_hours
        self.gap_days = gap_days

        # Calculate required gap according to Masters (2018), p. 17-18
        # Gap = min(lookback, lookahead) - 1
        self.required_gap_hours = min(lookback_hours, lookahead_hours) - 1
        self.actual_gap_hours = gap_days * 24  # Convert days to hours

        if self.actual_gap_hours < self.required_gap_hours:
            warnings.warn(
                f"Gap of {gap_days} days ({self.actual_gap_hours}h) is less than "
                f"required {self.required_gap_hours}h. This may cause future leak!"
            )

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load Bitcoin hourly candle data from CSV.

        Expected columns: timestamp, open, high, low, close, volume

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # Ensure we have OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by timestamp
        df = df.sort_index()

        # Check for missing hours (should be continuous hourly data)
        expected_hours = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='H'
        )
        missing_hours = expected_hours.difference(df.index)

        if len(missing_hours) > 0:
            warnings.warn(
                f"Found {len(missing_hours)} missing hours. "
                f"Forward filling missing values."
            )
            df = df.reindex(expected_hours, method='ffill')

        return df[required_cols]

    def split_into_years(
        self,
        df: pd.DataFrame,
        train_year: int,
        val_year: int,
        conf_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into three years with proper gaps to avoid overlap.

        According to Masters (2018), p. 16-18, we need gaps between datasets
        to prevent future leak when using lookback/lookahead windows.

        Args:
            df: Full DataFrame with hourly data
            train_year: Year to use for training (e.g., 2020)
            val_year: Year to use for validation (e.g., 2021)
            conf_year: Year to use for confidence intervals (e.g., 2022)

        Returns:
            Tuple of (train_df, val_df, conf_df)
        """
        # Define year boundaries
        train_start = pd.Timestamp(f'{train_year}-01-01')
        train_end = pd.Timestamp(f'{train_year}-12-31 23:59:59')

        val_start = pd.Timestamp(f'{val_year}-01-01')
        val_end = pd.Timestamp(f'{val_year}-12-31 23:59:59')

        conf_start = pd.Timestamp(f'{conf_year}-01-01')
        conf_end = pd.Timestamp(f'{conf_year}-12-31 23:59:59')

        # Apply gaps (shrink end of each set, expand start of next set)
        gap_timedelta = timedelta(hours=self.actual_gap_hours)

        # Training set: full year, but end earlier to leave gap
        train_end_adjusted = train_end - gap_timedelta
        train_df = df[train_start:train_end_adjusted].copy()

        # Validation set: start after gap, end before next gap
        val_start_adjusted = val_start + gap_timedelta
        val_end_adjusted = val_end - gap_timedelta
        val_df = df[val_start_adjusted:val_end_adjusted].copy()

        # Confidence set: start after gap, keep full end
        conf_start_adjusted = conf_start + gap_timedelta
        conf_df = df[conf_start_adjusted:conf_end].copy()

        print(f"\nDataset Split Summary:")
        print(f"  Training:   {train_start} to {train_end_adjusted} "
              f"({len(train_df)} hours, {len(train_df)/24:.1f} days)")
        print(f"  Gap:        {gap_timedelta} ({self.gap_days} days)")
        print(f"  Validation: {val_start_adjusted} to {val_end_adjusted} "
              f"({len(val_df)} hours, {len(val_df)/24:.1f} days)")
        print(f"  Gap:        {gap_timedelta} ({self.gap_days} days)")
        print(f"  Confidence: {conf_start_adjusted} to {conf_end} "
              f"({len(conf_df)} hours, {len(conf_df)/24:.1f} days)")

        return train_df, val_df, conf_df

    def create_sequences(
        self,
        df: pd.DataFrame,
        return_timestamps: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create input-output sequences for walk-forward testing.

        Each sequence uses lookback_hours as input to predict lookahead_hours.

        Args:
            df: DataFrame with OHLCV data
            return_timestamps: If True, also return timestamps for each sequence

        Returns:
            Tuple of (X, y, timestamps) where:
                X: Array of shape (n_sequences, lookback_hours, n_features)
                y: Array of shape (n_sequences, lookahead_hours, n_targets)
                timestamps: Array of timestamps for each sequence (if requested)
        """
        n_samples = len(df)
        sequence_length = self.lookback_hours + self.lookahead_hours

        # Check if we have enough data
        if n_samples < sequence_length:
            raise ValueError(
                f"Not enough data. Need at least {sequence_length} hours, "
                f"but got {n_samples}"
            )

        # Calculate number of sequences we can create
        n_sequences = n_samples - sequence_length + 1

        # Convert to numpy for faster processing
        data = df.values

        # Preallocate arrays
        X = np.zeros((n_sequences, self.lookback_hours, data.shape[1]))
        y = np.zeros((n_sequences, self.lookahead_hours, data.shape[1]))

        if return_timestamps:
            timestamps = np.zeros(n_sequences, dtype='datetime64[ns]')

        # Create sequences
        for i in range(n_sequences):
            X[i] = data[i:i + self.lookback_hours]
            y[i] = data[i + self.lookback_hours:i + sequence_length]

            if return_timestamps:
                timestamps[i] = df.index[i + self.lookback_hours]

        print(f"\nCreated {n_sequences} sequences:")
        print(f"  X shape: {X.shape} (sequences, lookback_hours, features)")
        print(f"  y shape: {y.shape} (sequences, lookahead_hours, targets)")

        if return_timestamps:
            return X, y, timestamps
        else:
            return X, y, None

    def get_daily_walk_forward_splits(
        self,
        df: pd.DataFrame
    ) -> list:
        """
        Generate daily walk-forward training/test splits.

        For each day in the dataset:
        - Training: Previous lookback_hours
        - Test: Next lookahead_hours

        Args:
            df: DataFrame with hourly data

        Returns:
            List of tuples (train_indices, test_indices, date)
        """
        splits = []

        # Start from first possible prediction point
        start_idx = self.lookback_hours

        # End when we can't make full predictions
        end_idx = len(df) - self.lookahead_hours

        # Generate daily splits (every 24 hours)
        for i in range(start_idx, end_idx, 24):
            train_indices = np.arange(i - self.lookback_hours, i)
            test_indices = np.arange(i, min(i + self.lookahead_hours, len(df)))

            date = df.index[i]
            splits.append((train_indices, test_indices, date))

        print(f"\nGenerated {len(splits)} daily walk-forward splits")
        print(f"  First split: {df.index[start_idx]}")
        print(f"  Last split:  {df.index[end_idx - 24]}")

        return splits


def get_feature_names() -> list:
    """
    Get standardized feature names for OHLCV data.

    Returns:
        List of feature names
    """
    return ['open', 'high', 'low', 'close', 'volume']


if __name__ == "__main__":
    # Example usage
    print("Bitcoin Data Loader - Example Usage")
    print("=" * 50)

    # Initialize loader
    loader = BitcoinDataLoader(
        lookback_hours=168,  # 7 days
        lookahead_hours=24,   # 1 day
        gap_days=1  # 1 day is sufficient (24h > 23h required)
    )

    print(f"\nConfiguration:")
    print(f"  Lookback:  {loader.lookback_hours} hours (7 days)")
    print(f"  Lookahead: {loader.lookahead_hours} hours (1 day)")
    print(f"  Gap:       {loader.gap_days} days ({loader.actual_gap_hours} hours)")
    print(f"  Required min gap: {loader.required_gap_hours} hours")
