"""
Cryptocurrency data loader for high-frequency (1-minute) candle data.
Supports Bitcoin and other cryptocurrencies via Binance API.
"""

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os
import pickle
from pathlib import Path


class CryptoDataLoader:
    """
    Load cryptocurrency OHLCV data at 1-minute resolution.

    Uses Binance public API (no authentication required).
    Handles rate limiting and automatic retry.
    """

    def __init__(self, cache_dir: str = "data/crypto_cache"):
        """
        Initialize the crypto data loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Binance API endpoints
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"

        # Rate limiting
        self.request_delay = 0.5  # Seconds between requests
        self.last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)

        self.last_request_time = time.time()

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch klines (candlestick) data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '1h', etc.)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max number of candles per request (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        url = f"{self.base_url}{self.klines_endpoint}"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            # Parse response
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert to appropriate types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def download_bitcoin_data(
        self,
        start_date: str = "2017-08-17",  # Bitcoin on Binance start date
        end_date: Optional[str] = None,
        interval: str = "1m",
        symbol: str = "BTCUSDT",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download Bitcoin data at specified interval.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Timeframe ('1m' for 1 minute)
            symbol: Trading pair (default: BTCUSDT)
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.pkl"

        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Downloading {symbol} data from {start_date} to {end_date}...")
        print(f"Interval: {interval}")

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        # Convert to milliseconds
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Fetch data in chunks (Binance limit: 1000 candles per request)
        all_data = []
        current_start = start_ms

        # Calculate interval in milliseconds
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
        }[interval]

        chunk_size = 1000  # Binance limit
        total_chunks = (end_ms - start_ms) // (interval_ms * chunk_size) + 1

        print(f"Estimated chunks: {total_chunks}")

        chunk_count = 0
        while current_start < end_ms:
            # Calculate chunk end
            chunk_end = min(current_start + interval_ms * chunk_size, end_ms)

            # Fetch chunk
            df_chunk = self._fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=chunk_end,
                limit=chunk_size
            )

            if df_chunk.empty:
                print(f"Warning: Empty chunk at {datetime.fromtimestamp(current_start/1000)}")
                break

            all_data.append(df_chunk)

            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"  Downloaded {chunk_count}/{total_chunks} chunks "
                      f"({len(all_data) * chunk_size:,} candles)...")

            # Move to next chunk
            current_start = int(df_chunk['timestamp'].iloc[-1].timestamp() * 1000) + interval_ms

        # Combine all chunks
        if not all_data:
            raise ValueError("No data downloaded!")

        df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        print(f"\nDownload complete!")
        print(f"  Total candles: {len(df):,}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"  Cached to {cache_file}")

        return df

    def download_all_available_bitcoin(
        self,
        interval: str = "1m",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download all available Bitcoin data from Binance.

        Args:
            interval: Timeframe ('1m' for 1 minute)
            use_cache: Use cached data if available

        Returns:
            DataFrame with all available OHLCV data
        """
        # Binance Bitcoin trading started on 2017-08-17
        start_date = "2017-08-17"
        end_date = datetime.now().strftime("%Y-%m-%d")

        return self.download_bitcoin_data(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache
        )

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
        Validate downloaded data for anomalies.

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, message)
        """
        issues = []

        # Check for missing values
        if df.isnull().any().any():
            issues.append(f"Missing values found: {df.isnull().sum().sum()}")

        # Check for invalid OHLC relationships
        invalid_high = (df['high'] < df['low']).sum()
        if invalid_high > 0:
            issues.append(f"Invalid high/low: {invalid_high} candles")

        invalid_oc = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
        if invalid_oc > 0:
            issues.append(f"Invalid open: {invalid_oc} candles")

        invalid_cc = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
        if invalid_cc > 0:
            issues.append(f"Invalid close: {invalid_cc} candles")

        # Check for extreme price movements (likely data errors)
        returns = df['close'].pct_change().abs()
        extreme_moves = (returns > 0.5).sum()  # >50% move in 1 minute
        if extreme_moves > 0:
            issues.append(f"Extreme price movements: {extreme_moves} candles")

        # Check for gaps in timestamps (if 1m data)
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        expected_diff = 60  # 1 minute in seconds
        gaps = (time_diffs > expected_diff * 2).sum()
        if gaps > 0:
            issues.append(f"Timestamp gaps: {gaps} gaps detected")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Data validation passed"

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing invalid candles and filling gaps.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        original_len = len(df)

        # Remove rows with missing values
        df = df.dropna()

        # Remove invalid OHLC relationships
        df = df[df['high'] >= df['low']]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]

        # Remove extreme outliers (>50% move in 1 minute)
        returns = df['close'].pct_change().abs()
        df = df[returns <= 0.5]

        removed = original_len - len(df)
        print(f"  Removed {removed} invalid candles ({removed/original_len*100:.2f}%)")

        return df.reset_index(drop=True)


if __name__ == "__main__":
    # Test data loader
    print("="*60)
    print("Testing Crypto Data Loader")
    print("="*60)

    loader = CryptoDataLoader()

    # Test with small date range first
    print("\nTest 1: Download 1 week of Bitcoin 1-minute data")
    start_date = "2024-01-01"
    end_date = "2024-01-08"

    df = loader.download_bitcoin_data(
        start_date=start_date,
        end_date=end_date,
        interval="1m",
        use_cache=False
    )

    print(f"\nData preview:")
    print(df.head(10))
    print(f"\nData info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Validate data
    print("\nValidating data...")
    is_valid, message = loader.validate_data(df)
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    # Clean data if needed
    if not is_valid:
        df = loader.clean_data(df)
        is_valid, message = loader.validate_data(df)
        print(f"\nAfter cleaning:")
        print(f"  Valid: {is_valid}")
        print(f"  Message: {message}")

    # Get OHLC array
    print("\nExtracting OHLC array...")
    ohlc = loader.get_ohlc_array(df)
    print(f"  OHLC shape: {ohlc.shape}")
    print(f"  OHLC sample (first 5 candles):")
    for i in range(5):
        print(f"    Candle {i}: O={ohlc[i,0]:.2f} H={ohlc[i,1]:.2f} "
              f"L={ohlc[i,2]:.2f} C={ohlc[i,3]:.2f}")

    print("\n" + "="*60)
    print("Data loader test completed successfully!")
    print("="*60)

    # Instructions for downloading full dataset
    print("\nTo download full Bitcoin history:")
    print("  loader = CryptoDataLoader()")
    print("  df = loader.download_all_available_bitcoin(interval='1m')")
    print("\nWarning: Full download may take 1-2 hours and use ~10GB disk space")
