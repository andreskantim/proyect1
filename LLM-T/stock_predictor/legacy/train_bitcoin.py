"""
Main training script for Market GPT on Bitcoin data.
Implements initial training phase with walk-forward capability.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from market_gpt import MarketGPT, MarketGPTConfig, create_market_gpt
from tokenizer import OHLCTokenizer, TokenizerConfig
from crypto_data_loader import CryptoDataLoader
from walk_forward_trainer import WalkForwardTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_directories(output_dir: str, log_dir: str):
    """Create necessary directories."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path("data/crypto_cache").mkdir(parents=True, exist_ok=True)


def main(args):
    print("="*80)
    print("MARKET GPT - BITCOIN TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    setup_directories(args.output_dir, args.log_dir)

    # Save config to output dir
    with open(Path(args.output_dir) / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ========================================
    # Step 1: Load Bitcoin Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: LOADING BITCOIN DATA")
    print("="*80)

    data_loader = CryptoDataLoader(cache_dir="data/crypto_cache")

    if config['data']['use_full_history']:
        print("Downloading all available Bitcoin history...")
        df = data_loader.download_all_available_bitcoin(
            interval=config['data']['interval'],
            use_cache=config['data']['use_cache']
        )
    else:
        print(f"Downloading Bitcoin data from {config['data']['start_date']} to {config['data']['end_date']}...")
        df = data_loader.download_bitcoin_data(
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            interval=config['data']['interval'],
            use_cache=config['data']['use_cache']
        )

    # Validate and clean data
    is_valid, message = data_loader.validate_data(df)
    print(f"\nData validation: {message}")

    if not is_valid:
        print("Cleaning data...")
        df = data_loader.clean_data(df)

    # Extract OHLC
    ohlc_data = data_loader.get_ohlc_array(df)
    print(f"\nOHLC data shape: {ohlc_data.shape}")
    print(f"Price range: [{ohlc_data.min():.2f}, {ohlc_data.max():.2f}]")

    # Split train/val/test
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data'].get('val_ratio', 0.1)

    n_samples = len(ohlc_data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_data = ohlc_data[:train_end]
    val_data = ohlc_data[train_end:val_end]
    test_data = ohlc_data[val_end:]

    print(f"\nData splits:")
    print(f"  Train: {len(train_data):,} candles")
    print(f"  Val:   {len(val_data):,} candles")
    print(f"  Test:  {len(test_data):,} candles")

    # Save data splits info
    data_info = {
        'total_samples': n_samples,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'train_ratio': train_ratio,
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        },
        'price_stats': {
            'min': float(ohlc_data.min()),
            'max': float(ohlc_data.max()),
            'mean': float(ohlc_data.mean()),
            'std': float(ohlc_data.std())
        }
    }

    with open(Path(args.output_dir) / "data_info.json", 'w') as f:
        json.dump(data_info, f, indent=2)

    # ========================================
    # Step 2: Create and Fit Tokenizer
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: TOKENIZER")
    print("="*80)

    tokenizer_config = TokenizerConfig(
        vocab_size=config['tokenizer']['vocab_size'],
        n_features=4
    )

    tokenizer = OHLCTokenizer(tokenizer_config)
    tokenizer.fit(train_data)

    # Save tokenizer
    tokenizer.save(Path(args.output_dir) / "tokenizer.pkl")

    # Print vocabulary statistics
    vocab_stats = tokenizer.get_vocab_stats()
    print("\nVocabulary statistics:")
    for key, value in vocab_stats.items():
        print(f"  {key}: {value}")

    # ========================================
    # Step 3: Create Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: MODEL CREATION")
    print("="*80)

    model_config = MarketGPTConfig(
        n_layers=config['model']['n_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['context_length'],
        vocab_size=config['tokenizer']['vocab_size'],
        n_features=4,
        n_steps_pred=config['model'].get('n_steps_pred', 10),
        dropout=config['model']['dropout'],
        gradient_checkpointing=args.gradient_checkpointing
    )

    model = MarketGPT(model_config)

    # Print model info
    param_counts = model.count_parameters()
    print(f"\nModel architecture:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Model dim: {model_config.d_model}")
    print(f"  Attention heads: {model_config.n_heads}")
    print(f"  Context length: {model_config.context_length}")
    print(f"  Parameters: {param_counts['total']:,}")

    # Enable mixed precision if requested
    if args.mixed_precision:
        print("\nMixed precision training enabled (FP16)")

    # ========================================
    # Step 4: Initial Training
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: INITIAL TRAINING")
    print("="*80)

    trainer = WalkForwardTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )

    # Train
    train_config = config['training']['initial']
    history = trainer.initial_training(
        train_data=train_data,
        val_split=config['data'].get('val_ratio', 0.1) / config['data']['train_ratio'],  # Adjust for train split
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        learning_rate=train_config['learning_rate'],
        warmup_steps=train_config['warmup_steps'],
        save_dir=args.output_dir
    )

    # Save training history
    with open(Path(args.output_dir) / "train_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Model saved to: {args.output_dir}/best_initial.pt")
    print(f"Tokenizer saved to: {args.output_dir}/tokenizer.pkl")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Market GPT on Bitcoin data")

    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save model and results')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory to save logs')

    # Device options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    main(args)
