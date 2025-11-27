"""
Gold data loader for high-frequency data.
Supports multiple sources and interpolation to 1-minute resolution.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class GoldDataLoader:
    """
    Load gold (XAU/USD) data at various resolutions.

    Note: 1-minute gold data for 20 years is rarely available from free APIs.
    This loader will:
    1. Attempt to get highest resolution available (5min, 15min, hourly)
    2. Interpolate to 1-minute if needed
    3. Cache results for faster access
    """

    def __init__(self, cache_dir: str = "data/gold_cache"):
        """
        Initialize the gold data loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_gold_data_yfinance(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download gold data using yfinance (Yahoo Finance).

        Available intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        Note: 1m data only available for last 7 days

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            interval: Data interval
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_file = self.cache_dir / f"GC_F_{interval}_{start_date}_{end_date}.pkl"

        if use_cache and cache_file.exists():
            print(f"Loading cached gold data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Downloading gold data from {start_date} to {end_date}...")
        print(f"  Interval: {interval}")
        print(f"  Using yfinance (Yahoo Finance)")

        # Download using yfinance
        # GC=F is gold futures, XAU-USD is spot gold
        ticker = yf.Ticker("GC=F")  # Gold futures

        try:
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )

            if df.empty:
                print("  Warning: yfinance returned empty data, trying XAU-USD...")
                ticker = yf.Ticker("XAUUSD=X")
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True
                )

            if df.empty:
                raise ValueError("No data available from yfinance")

            # Rename columns to match our format
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have timestamp column
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                df['timestamp'] = df.index

            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            print(f"  Downloaded {len(df):,} candles")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"  Cached to {cache_file}")

            return df

        except Exception as e:
            print(f"Error downloading gold data: {e}")
            raise

    def download_gold_20_years(
        self,
        end_date: Optional[str] = None,
        interval: str = "1h",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download 20 years of gold data.

        Args:
            end_date: End date (default: today)
            interval: Data interval (best resolution: 1h for 20 years)
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Calculate start date (20 years back)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=20*365 + 5)  # +5 for leap years
        start_date = start_dt.strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"Downloading 20 years of gold data")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")

        # yfinance has limitations on data range vs interval
        # For 20 years, we need to use hourly or daily data
        if interval in ['1m', '2m', '5m', '15m', '30m']:
            print(f"  Warning: {interval} not available for 20 years")
            print(f"  Using 1h interval instead (will interpolate if needed)")
            interval = '1h'

        return self.download_gold_data_yfinance(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache
        )

    def interpolate_to_1min(
        self,
        df: pd.DataFrame,
        method: str = "linear"
    ) -> pd.DataFrame:
        """
        Interpolate data to 1-minute resolution.

        Args:
            df: DataFrame with OHLCV data at any resolution
            method: Interpolation method ('linear', 'cubic', 'akima')

        Returns:
            DataFrame with 1-minute candles
        """
        print(f"\nInterpolating to 1-minute resolution...")
        print(f"  Original: {len(df):,} candles")

        # Create complete 1-minute time range
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()

        # Create 1-minute index
        minute_index = pd.date_range(start=start_time, end=end_time, freq='1min')

        # Set timestamp as index
        df_indexed = df.set_index('timestamp')

        # Reindex to 1-minute intervals
        df_reindexed = df_indexed.reindex(minute_index)

        # Interpolate OHLC
        # For OHLC data, we use different strategies:
        # - Open/Close: linear interpolation
        # - High: forward fill then interpolate (preserve highs)
        # - Low: forward fill then interpolate (preserve lows)

        if method == "linear":
            df_reindexed['open'] = df_reindexed['open'].interpolate(method='linear')
            df_reindexed['close'] = df_reindexed['close'].interpolate(method='linear')
            df_reindexed['high'] = df_reindexed['high'].interpolate(method='linear')
            df_reindexed['low'] = df_reindexed['low'].interpolate(method='linear')
        elif method == "cubic":
            df_reindexed['open'] = df_reindexed['open'].interpolate(method='cubic')
            df_reindexed['close'] = df_reindexed['close'].interpolate(method='cubic')
            df_reindexed['high'] = df_reindexed['high'].interpolate(method='cubic')
            df_reindexed['low'] = df_reindexed['low'].interpolate(method='cubic')
        else:
            df_reindexed = df_reindexed.interpolate(method=method)

        # Volume: forward fill
        df_reindexed['volume'] = df_reindexed['volume'].fillna(method='ffill')

        # Reset index
        df_reindexed = df_reindexed.reset_index()
        df_reindexed.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Ensure OHLC relationships are valid after interpolation
        df_reindexed['high'] = df_reindexed[['open', 'high', 'close']].max(axis=1)
        df_reindexed['low'] = df_reindexed[['open', 'low', 'close']].min(axis=1)

        print(f"  Interpolated: {len(df_reindexed):,} candles")
        print(f"  Expansion factor: {len(df_reindexed) / len(df):.1f}x")

        return df_reindexed

    def get_ohlc_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract OHLC values as numpy array.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            OHLC array of shape (n_samples, 4)
        """
        return df[['open', 'high', 'low', 'close']].values

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data for anomalies.

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, message)
        """
        issues = []

        # Check for missing values
        if df.isnull().any().any():
            issues.append(f"Missing values: {df.isnull().sum().sum()}")

        # Check OHLC relationships
        invalid_high = (df['high'] < df['low']).sum()
        if invalid_high > 0:
            issues.append(f"Invalid high/low: {invalid_high}")

        invalid_oc = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
        if invalid_oc > 0:
            issues.append(f"Invalid open: {invalid_oc}")

        invalid_cc = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
        if invalid_cc > 0:
            issues.append(f"Invalid close: {invalid_cc}")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Data validation passed"

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing invalid entries.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning gold data...")
        original_len = len(df)

        # Remove rows with missing values
        df = df.dropna()

        # Fix OHLC relationships
        df = df[df['high'] >= df['low']]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]

        removed = original_len - len(df)
        print(f"  Removed {removed} invalid candles ({removed/original_len*100:.2f}%)")

        return df.reset_index(drop=True)


if __name__ == "__main__":
    # Test gold data loader
    print("="*60)
    print("Testing Gold Data Loader")
    print("="*60)

    loader = GoldDataLoader()

    # Test 1: Download 1 year of hourly data
    print("\nTest 1: Download 1 year of hourly gold data")
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    df = loader.download_gold_data_yfinance(
        start_date=start_date,
        end_date=end_date,
        interval="1h",
        use_cache=False
    )

    print(f"\nData preview:")
    print(df.head())
    print(f"\nData info:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Validate
    is_valid, message = loader.validate_data(df)
    print(f"\nValidation: {is_valid} - {message}")

    if not is_valid:
        df = loader.clean_data(df)

    # Test 2: Interpolate to 1-minute
    print("\nTest 2: Interpolate to 1-minute resolution")
    df_1min = loader.interpolate_to_1min(df.head(100))  # Test on small subset

    print(f"\nInterpolated data preview:")
    print(df_1min.head())
    print(f"  Shape: {df_1min.shape}")

    # Get OHLC array
    ohlc = loader.get_ohlc_array(df_1min)
    print(f"\nOHLC array shape: {ohlc.shape}")
    print(f"OHLC sample (first 5 candles):")
    for i in range(5):
        print(f"  Candle {i}: O={ohlc[i,0]:.2f} H={ohlc[i,1]:.2f} "
              f"L={ohlc[i,2]:.2f} C={ohlc[i,3]:.2f}")

    print("\n" + "="*60)
    print("Gold data loader test completed!")
    print("="*60)

    print("\nNote: To download full 20 years of gold data:")
    print("  loader = GoldDataLoader()")
    print("  df = loader.download_gold_20_years(interval='1h')")
    print("  df_1min = loader.interpolate_to_1min(df)")
    print("\nWarning: 20 years at 1-minute resolution = ~10M candles")
    print("This will require significant disk space and processing time")
