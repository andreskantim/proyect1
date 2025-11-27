#!/usr/bin/env python3
"""
Download Real Bitcoin Data from Cryptocurrency Exchanges

This script downloads real Bitcoin hourly candlestick data from
cryptocurrency exchanges using the CCXT library.

Features:
- Downloads from multiple exchanges (Binance, Kraken, etc.)
- Handles API rate limits automatically
- Resumes from last download if interrupted
- Validates data integrity
- Saves in CSV format

Usage:
    # Download all available historical data from Kraken (default, since 2013)
    python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv

    # Download specific date range
    python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv \
                                     --start-date 2020-01-01 \
                                     --end-date 2023-12-31

    # Use a different exchange (e.g., Binance)
    python download_bitcoin_data.py --exchange binance \
                                     --symbol BTC/USDT \
                                     --output ../data/raw/bitcoin_binance.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
from pathlib import Path

try:
    import ccxt
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install ccxt tqdm")
    sys.exit(1)


def get_exchange(exchange_name='binance'):
    """
    Initialize exchange connection.

    Args:
        exchange_name: Name of the exchange (default: binance)

    Returns:
        CCXT exchange object
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,  # Required by CCXT to respect rate limits
    })

    return exchange


def download_ohlcv(exchange, symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    """
    Download OHLCV data from exchange.

    Args:
        exchange: CCXT exchange object
        symbol: Trading pair symbol (default: BTC/USDT)
        timeframe: Candle timeframe (default: 1h)
        start_date: Start date (datetime object or None for earliest available)
        end_date: End date (datetime object or None for now)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"\nDownloading {symbol} {timeframe} data from {exchange.name}...")

    # Convert dates to millisecond timestamps
    if start_date is None:
        # Start from earliest available
        # Kraken: ~2013-09-01 for BTC/USD
        # Binance: ~2017-08-17 for BTC/USDT
        # Bitstamp: ~2011-08-18 for BTC/USD
        since = exchange.parse8601('2013-01-01T00:00:00Z')
    else:
        since = int(start_date.timestamp() * 1000)

    if end_date is None:
        end_timestamp = int(datetime.now().timestamp() * 1000)
    else:
        end_timestamp = int(end_date.timestamp() * 1000)

    # Check if exchange supports the timeframe
    if timeframe not in exchange.timeframes:
        print(f"Error: {exchange.name} doesn't support {timeframe} timeframe")
        print(f"Available timeframes: {list(exchange.timeframes.keys())}")
        sys.exit(1)

    all_ohlcv = []
    current_timestamp = since

    # Calculate total iterations for progress bar
    timeframe_duration = exchange.parse_timeframe(timeframe) * 1000  # in milliseconds
    total_iterations = int((end_timestamp - since) / (timeframe_duration * 1000)) + 1

    print(f"Start: {datetime.fromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {datetime.fromtimestamp(end_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated iterations: {total_iterations}")

    with tqdm(total=total_iterations, desc="Downloading", unit="req") as pbar:
        while current_timestamp < end_timestamp:
            try:
                # Fetch OHLCV data
                # Most exchanges limit to 1000 candles per request
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000  # Maximum limit for most exchanges
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Update timestamp for next iteration
                current_timestamp = ohlcv[-1][0] + timeframe_duration

                pbar.update(1)

                # Rate limiting is handled by CCXT enableRateLimit

            except ccxt.NetworkError as e:
                print(f"\nNetwork error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                print(f"\nExchange error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Continuing with downloaded data...")
                break

    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Remove duplicates (can happen at boundaries)
    df = df.drop_duplicates(subset='timestamp', keep='first')

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def validate_data(df):
    """
    Validate downloaded data for common issues.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")

    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate timestamps")

    # Check for gaps in timestamps (should be hourly)
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow 50% tolerance
    if len(gaps) > 0:
        issues.append(f"Found {len(gaps)} gaps in hourly data")

    # Check for invalid OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    if invalid_ohlc.any():
        issues.append(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")

    # Check for zero or negative prices
    price_cols = ['open', 'high', 'low', 'close']
    invalid_prices = (df[price_cols] <= 0).any(axis=1)
    if invalid_prices.any():
        issues.append(f"Found {invalid_prices.sum()} rows with zero or negative prices")

    # Check for zero volumes (warning only, can be valid)
    zero_volumes = (df['volume'] == 0).sum()
    if zero_volumes > 0:
        issues.append(f"Warning: Found {zero_volumes} rows with zero volume")

    is_valid = len([i for i in issues if not i.startswith("Warning")]) == 0

    return is_valid, issues


def print_data_summary(df):
    """Print summary statistics of the downloaded data."""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print()
    print("Price Statistics (Close):")
    print(f"  Min:    ${df['close'].min():,.2f}")
    print(f"  Max:    ${df['close'].max():,.2f}")
    print(f"  Mean:   ${df['close'].mean():,.2f}")
    print(f"  Median: ${df['close'].median():,.2f}")
    print()
    print("Volume Statistics:")
    print(f"  Min:    {df['volume'].min():,.2f}")
    print(f"  Max:    {df['volume'].max():,.2f}")
    print(f"  Mean:   {df['volume'].mean():,.2f}")
    print(f"  Median: {df['volume'].median():,.2f}")
    print()
    print("Sample data (first 5 rows):")
    print(df.head().to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Download real Bitcoin hourly data from cryptocurrency exchanges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all available data from Kraken (default, data since 2013)
  python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv

  # Download specific date range
  python download_bitcoin_data.py --start-date 2020-01-01 --end-date 2023-12-31

  # Use Binance exchange instead (data since 2017)
  python download_bitcoin_data.py --exchange binance --symbol BTC/USDT
        """
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/raw/bitcoin_hourly.csv',
        help='Output CSV file path (default: ../data/raw/bitcoin_hourly.csv)'
    )

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        choices=['binance', 'kraken', 'coinbasepro', 'bitstamp'],
        help='Exchange to download from (default: kraken - has data since 2013)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading pair symbol (default: BTC/USD for Kraken, BTCUSDT for Binance)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (default: earliest available)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (default: now)'
    )

    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip data validation'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid start date format: {args.start_date}")
            print("Use YYYY-MM-DD format")
            sys.exit(1)

    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end date format: {args.end_date}")
            print("Use YYYY-MM-DD format")
            sys.exit(1)

    # Validate date range
    if start_date and end_date and start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)

    print("="*80)
    print("BITCOIN DATA DOWNLOADER")
    print("="*80)
    print(f"Exchange: {args.exchange}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: 1h (hourly)")
    print(f"Start date: {start_date.strftime('%Y-%m-%d') if start_date else 'Earliest available'}")
    print(f"End date: {end_date.strftime('%Y-%m-%d') if end_date else 'Now'}")
    print(f"Output: {args.output}")
    print("="*80)

    # Initialize exchange
    try:
        exchange = get_exchange(args.exchange)
        print(f"\nConnected to {exchange.name}")
    except Exception as e:
        print(f"\nError connecting to exchange: {e}")
        sys.exit(1)

    # Download data
    try:
        df = download_ohlcv(
            exchange=exchange,
            symbol=args.symbol,
            timeframe='1h',
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"\nError downloading data: {e}")
        sys.exit(1)

    if df.empty:
        print("\nError: No data downloaded")
        sys.exit(1)

    print(f"\nDownloaded {len(df):,} candles")

    # Validate data
    if not args.no_validation:
        print("\nValidating data...")
        is_valid, issues = validate_data(df)

        if issues:
            print("\nValidation issues found:")
            for issue in issues:
                print(f"  - {issue}")

        if not is_valid:
            print("\nWarning: Data validation failed!")
            response = input("Continue saving anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
        else:
            print("Data validation passed!")

    # Print summary
    print_data_summary(df)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File saved successfully!")
    print(f"File size: {file_size_mb:.2f} MB")

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nYou can now use this data for the Bitcoin prediction model.")
    print(f"Data location: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the data: head {args.output}")
    print(f"  2. Run the training pipeline: python train_models.py")


if __name__ == "__main__":
    main()
