#!/usr/bin/env python3
"""
Generate Sample Bitcoin Data for Testing

This script generates synthetic Bitcoin hourly candlestick data
that can be used for testing the prediction pipeline.

Usage:
    python generate_sample_data.py --output ../data/raw/bitcoin_sample.csv \
                                    --years 3 \
                                    --seed 42
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_bitcoin_data(start_date, n_hours, initial_price=10000, seed=42):
    """
    Generate synthetic Bitcoin hourly candlestick data.

    Uses a geometric Brownian motion with some realistic characteristics:
    - Volatility clustering
    - Trending behavior
    - Volume correlation with price changes

    Args:
        start_date: Start date for data
        n_hours: Number of hours to generate
        initial_price: Initial Bitcoin price
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Parameters
    mu = 0.00005  # Drift (slight upward trend)
    sigma = 0.02  # Volatility

    # Generate price path
    returns = np.random.normal(mu, sigma, n_hours)

    # Add volatility clustering (GARCH-like behavior)
    volatility = np.ones(n_hours) * sigma
    for i in range(1, n_hours):
        volatility[i] = 0.05 * abs(returns[i-1]) + 0.9 * volatility[i-1]
        returns[i] = np.random.normal(mu, volatility[i])

    # Calculate prices
    price_multipliers = np.exp(np.cumsum(returns))
    close_prices = initial_price * price_multipliers

    # Generate OHLC from close prices
    data = []
    timestamps = [start_date + timedelta(hours=i) for i in range(n_hours)]

    for i in range(n_hours):
        close = close_prices[i]

        # Generate open, high, low with realistic spreads
        spread_pct = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5%

        open_price = close * (1 + np.random.normal(0, spread_pct))
        high = max(open_price, close) * (1 + np.random.uniform(0, spread_pct))
        low = min(open_price, close) * (1 - np.random.uniform(0, spread_pct))

        # Generate volume (correlated with price changes)
        base_volume = 1000000  # Base volume
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)

        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Bitcoin data for testing'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/raw/bitcoin_sample.csv',
        help='Output CSV file path'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Number of years of data to generate (default: 3)'
    )

    parser.add_argument(
        '--start_date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--initial_price',
        type=float,
        default=10000.0,
        help='Initial Bitcoin price'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Calculate number of hours
    n_hours = args.years * 365 * 24 + 24  # +24 for safety margin

    # Parse start date
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    print("=" * 60)
    print("Generating Synthetic Bitcoin Data")
    print("=" * 60)
    print(f"Start date: {start_date}")
    print(f"Years: {args.years}")
    print(f"Hours: {n_hours:,}")
    print(f"Initial price: ${args.initial_price:,.2f}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Generate data
    print("\nGenerating data...")
    df = generate_bitcoin_data(
        start_date=start_date,
        n_hours=n_hours,
        initial_price=args.initial_price,
        seed=args.seed
    )

    # Print statistics
    print("\nData Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():,.2f} to ${df['close'].max():,.2f}")
    print(f"  Mean price: ${df['close'].mean():,.2f}")
    print(f"  Mean volume: {df['volume'].mean():,.0f}")

    # Save to CSV
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nData saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
