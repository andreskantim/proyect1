"""
Multi-Market Data Loader for Case B: Reduced (100 assets)

Downloads and processes data from multiple markets:
- US Stocks: 50 assets
- Crypto: 20 assets
- Commodities: 15 assets
- Emerging Markets: 15 assets

Total: 100 assets across 4 categories
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
from tqdm import tqdm


class MultiMarketLoader:
    """
    Loads 100 assets from multiple markets for Case B.
    """

    # Category 0: US Stocks (50 assets - top SP500)
    US_STOCKS = [
        # Mega caps
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        # Large caps
        'V', 'WMT', 'JPM', 'MA', 'PG', 'XOM', 'HD', 'CVX', 'MRK', 'ABBV',
        'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'DHR', 'ABT',
        # Mid-large caps
        'NKE', 'TXN', 'NEE', 'DIS', 'VZ', 'CMCSA', 'ADBE', 'PM', 'NFLX', 'CRM',
        # Growth + Value mix
        'INTC', 'AMD', 'QCOM', 'HON', 'UNP', 'LOW', 'IBM', 'BA', 'CAT', 'GE'
    ]

    # Category 1: Crypto (20 assets)
    CRYPTO = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'DOGE-USD', 'SOL-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD',
        'LTC-USD', 'SHIB-USD', 'AVAX-USD', 'UNI-USD', 'LINK-USD',
        'ATOM-USD', 'XLM-USD', 'BCH-USD', 'FIL-USD', 'APT-USD'
    ]

    # Category 2: Commodities (15 assets)
    COMMODITIES = [
        'GC=F',   # Gold
        'SI=F',   # Silver
        'CL=F',   # Crude Oil WTI
        'BZ=F',   # Brent Oil
        'NG=F',   # Natural Gas
        'HG=F',   # Copper
        'PL=F',   # Platinum
        'PA=F',   # Palladium
        'ZC=F',   # Corn
        'ZS=F',   # Soybeans
        'ZW=F',   # Wheat
        'KC=F',   # Coffee
        'SB=F',   # Sugar
        'CC=F',   # Cocoa
        'CT=F',   # Cotton
    ]

    # Category 3: Emerging Markets (15 assets - ETFs and indices)
    EMERGING_MARKETS = [
        'EEM',    # iShares MSCI Emerging Markets ETF
        'VWO',    # Vanguard FTSE Emerging Markets ETF
        'IEMG',   # iShares Core MSCI Emerging Markets ETF
        'EWZ',    # Brazil
        'FXI',    # China Large Cap
        'INDA',   # India
        'RSX',    # Russia (if available)
        'EWY',    # South Korea
        'EWT',    # Taiwan
        'EWW',    # Mexico
        'EZA',    # South Africa
        'THD',    # Thailand
        'EIDO',   # Indonesia
        'EPHE',   # Philippines
        'TUR',    # Turkey
    ]

    CATEGORIES = {
        0: "US_Stocks",
        1: "Crypto",
        2: "Commodities",
        3: "Emerging_Markets"
    }

    def __init__(self, cache_dir: str = "../data/reduced_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build asset mappings
        self.all_assets = []
        self.asset_to_id = {}
        self.asset_to_category = {}

        asset_id = 0
        for category_id, assets in enumerate([
            self.US_STOCKS,
            self.CRYPTO,
            self.COMMODITIES,
            self.EMERGING_MARKETS
        ]):
            for symbol in assets:
                self.all_assets.append(symbol)
                self.asset_to_id[symbol] = asset_id
                self.asset_to_category[symbol] = category_id
                asset_id += 1

        print(f"Initialized MultiMarketLoader with {len(self.all_assets)} assets")
        print(f"Categories: {self.CATEGORIES}")

    def download_all_assets(
        self,
        start_date: str = "2014-01-01",  # 10 years
        end_date: str = None,
        force_redownload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all 100 assets.

        Returns:
            Dictionary mapping asset symbol to DataFrame
        """
        print(f"\nDownloading 100 assets from {start_date} to {end_date or 'today'}")
        print("=" * 60)

        all_data = {}
        failed_assets = []

        for symbol in tqdm(self.all_assets, desc="Downloading assets"):
            cache_file = self.cache_dir / f"{symbol.replace('=', '_').replace('-', '_')}.csv"

            # Check cache
            if cache_file.exists() and not force_redownload:
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    all_data[symbol] = df
                    continue
                except Exception as e:
                    print(f"Cache read failed for {symbol}: {e}")

            # Download
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    auto_adjust=True
                )

                if len(df) < 100:  # Minimum data requirement
                    print(f"Insufficient data for {symbol}: {len(df)} rows")
                    failed_assets.append(symbol)
                    continue

                # Keep only OHLCV
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                # Add metadata
                df['asset_id'] = self.asset_to_id[symbol]
                df['category_id'] = self.asset_to_category[symbol]
                df['symbol'] = symbol

                # Cache
                df.to_csv(cache_file)
                all_data[symbol] = df

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
                failed_assets.append(symbol)

        print(f"\nDownload complete:")
        print(f"  Successfully downloaded: {len(all_data)}/{len(self.all_assets)}")
        if failed_assets:
            print(f"  Failed assets: {failed_assets}")

        return all_data

    def prepare_training_data(
        self,
        all_data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation, and test sets.
        Uses temporal split (NOT random) to prevent data leakage.
        """
        # Concatenate all data
        combined = pd.concat(all_data.values(), axis=0)
        combined = combined.sort_index()

        print(f"\nCombined dataset:")
        print(f"  Total candles: {len(combined):,}")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"  Assets: {combined['asset_id'].nunique()}")
        print(f"  Categories: {combined['category_id'].nunique()}")

        # Temporal split
        n = len(combined)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = combined.iloc[:train_end].copy()
        val_df = combined.iloc[train_end:val_end].copy()
        test_df = combined.iloc[val_end:].copy()

        print(f"\nSplit:")
        print(f"  Train: {len(train_df):,} ({100*train_ratio:.1f}%)")
        print(f"  Val:   {len(val_df):,} ({100*val_ratio:.1f}%)")
        print(f"  Test:  {len(test_df):,} ({100*(1-train_ratio-val_ratio):.1f}%)")

        return train_df, val_df, test_df

    def get_category_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics per category."""
        stats = df.groupby('category_id').agg({
            'Close': 'count',
            'asset_id': 'nunique'
        }).rename(columns={'Close': 'num_candles', 'asset_id': 'num_assets'})

        stats['category_name'] = stats.index.map(self.CATEGORIES)
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = MultiMarketLoader()

    print("\nDownloading data...")
    all_data = loader.download_all_assets(
        start_date="2014-01-01",
        force_redownload=False
    )

    print("\nPreparing splits...")
    train_df, val_df, test_df = loader.prepare_training_data(all_data)

    print("\nCategory statistics (training set):")
    print(loader.get_category_stats(train_df))

    print("\nData loading test complete!")
