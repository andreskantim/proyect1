"""
Crypto Multi-Asset Data Loader - CASE C (Prototype)
Loads 20 top cryptocurrencies with daily candles.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
from tqdm import tqdm


class CryptoMultiAssetLoader:
    """
    Load multiple cryptocurrencies for multi-asset training.
    Case C: 20 top cryptos, daily candles, ~5-8 years.
    """

    # Top 20 cryptocurrencies by market cap
    CRYPTO_ASSETS = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'BNB-USD',   # Binance Coin
        'XRP-USD',   # Ripple
        'ADA-USD',   # Cardano
        'DOGE-USD',  # Dogecoin
        'SOL-USD',   # Solana
        'TRX-USD',   # Tron
        'DOT-USD',   # Polkadot
        'MATIC-USD', # Polygon
        'LTC-USD',   # Litecoin
        'SHIB-USD',  # Shiba Inu
        'AVAX-USD',  # Avalanche
        'UNI-USD',   # Uniswap
        'LINK-USD',  # Chainlink
        'ATOM-USD',  # Cosmos
        'XLM-USD',   # Stellar
        'BCH-USD',   # Bitcoin Cash
        'FIL-USD',   # Filecoin
        'APT-USD',   # Aptos
    ]

    CATEGORY_ID = 4  # Crypto category

    def __init__(self, cache_dir: str = "data/crypto_multi_cache"):
        """Initialize loader with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Asset to ID mapping
        self.asset_to_id = {asset: i for i, asset in enumerate(self.CRYPTO_ASSETS)}
        self.id_to_asset = {i: asset for asset, i in self.asset_to_id.items()}

    def download_all_cryptos(
        self,
        start_date: str = "2017-01-01",
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download all crypto assets.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date (default: today)
            use_cache: Use cached data if available

        Returns:
            DataFrame with columns: [timestamp, asset_id, category_id, open, high, low, close, volume]
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_file = self.cache_dir / f"crypto_multi_{start_date}_{end_date}.pkl"

        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)

        print(f"\n{'='*70}")
        print(f"Downloading 20 Cryptocurrencies (Daily Candles)")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*70}\n")

        all_data = []

        for asset_symbol in tqdm(self.CRYPTO_ASSETS, desc="Downloading cryptos"):
            try:
                # Download using yfinance
                ticker = yf.Ticker(asset_symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)

                if df.empty:
                    print(f"  Warning: No data for {asset_symbol}")
                    continue

                # Reset index and clean
                df = df.reset_index()
                df.columns = [col.lower() for col in df.columns]

                # Rename date/datetime to timestamp
                if 'datetime' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['datetime'])
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])

                # Add asset info
                df['asset'] = asset_symbol
                df['asset_id'] = self.asset_to_id[asset_symbol]
                df['category_id'] = self.CATEGORY_ID

                # Keep only OHLC + metadata
                df = df[['timestamp', 'asset', 'asset_id', 'category_id',
                        'open', 'high', 'low', 'close', 'volume']]

                all_data.append(df)

                print(f"  ✓ {asset_symbol}: {len(df)} days")

            except Exception as e:
                print(f"  ✗ Error downloading {asset_symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No crypto data downloaded!")

        # Combine all assets
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['asset_id', 'timestamp']).reset_index(drop=True)

        # Clean NaN values in OHLC columns
        initial_rows = len(combined_df)
        combined_df = combined_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        dropped_rows = initial_rows - len(combined_df)

        if dropped_rows > 0:
            print(f"\n⚠️  Dropped {dropped_rows:,} rows with NaN values ({dropped_rows/initial_rows*100:.2f}%)")

        print(f"\n{'='*70}")
        print(f"Download Complete!")
        print(f"  Total assets: {len(self.CRYPTO_ASSETS)}")
        print(f"  Total candles: {len(combined_df):,}")
        print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"  Assets with data: {combined_df['asset'].nunique()}")
        print(f"{'='*70}\n")

        # Cache
        combined_df.to_pickle(cache_file)
        print(f"Cached to: {cache_file}\n")

        return combined_df

    def get_ohlc_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract OHLC values as numpy array.

        Args:
            df: DataFrame with OHLC data

        Returns:
            OHLC array of shape (n_samples, 4)
        """
        return df[['open', 'high', 'low', 'close']].values

    def get_asset_ids(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract asset IDs.

        Args:
            df: DataFrame with asset_id column

        Returns:
            Asset IDs array of shape (n_samples,)
        """
        return df['asset_id'].values

    def get_category_ids(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract category IDs.

        Args:
            df: DataFrame with category_id column

        Returns:
            Category IDs array of shape (n_samples,)
        """
        return df['category_id'].values

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets temporally.

        Args:
            df: Combined DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            (train_df, val_df, test_df)
        """
        # Get unique timestamps (sorted)
        timestamps = sorted(df['timestamp'].unique())

        n_timestamps = len(timestamps)
        train_end = int(n_timestamps * train_ratio)
        val_end = int(n_timestamps * (train_ratio + val_ratio))

        train_timestamps = timestamps[:train_end]
        val_timestamps = timestamps[train_end:val_end]
        test_timestamps = timestamps[val_end:]

        train_df = df[df['timestamp'].isin(train_timestamps)].copy()
        val_df = df[df['timestamp'].isin(val_timestamps)].copy()
        test_df = df[df['timestamp'].isin(test_timestamps)].copy()

        print(f"Data split:")
        print(f"  Train: {len(train_df):,} candles ({len(train_timestamps)} days)")
        print(f"  Val:   {len(val_df):,} candles ({len(val_timestamps)} days)")
        print(f"  Test:  {len(test_df):,} candles ({len(test_timestamps)} days)")

        return train_df, val_df, test_df

    def get_data_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the dataset."""
        return {
            'total_candles': len(df),
            'num_assets': df['asset'].nunique(),
            'num_categories': df['category_id'].nunique(),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max()),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'price_stats': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'mean': float(df['close'].mean()),
                'median': float(df['close'].median())
            },
            'assets': df.groupby('asset').size().to_dict()
        }


if __name__ == "__main__":
    # Test
    print("="*70)
    print("Testing Crypto Multi-Asset Data Loader")
    print("="*70)

    loader = CryptoMultiAssetLoader()

    # Download all cryptos
    df = loader.download_all_cryptos(
        start_date="2020-01-01",
        end_date="2024-01-01",
        use_cache=False
    )

    # Get stats
    stats = loader.get_data_stats(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Split data
    train_df, val_df, test_df = loader.split_train_val_test(df)

    # Extract arrays
    train_ohlc = loader.get_ohlc_array(train_df)
    train_asset_ids = loader.get_asset_ids(train_df)

    print(f"\nArray shapes:")
    print(f"  OHLC: {train_ohlc.shape}")
    print(f"  Asset IDs: {train_asset_ids.shape}")

    print("\n" + "="*70)
    print("Data loader test completed successfully!")
    print("="*70)
