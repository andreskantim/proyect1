"""
Training script for Case C: Crypto Prototype
20 cryptocurrencies, daily candles, multi-asset model.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent and common to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.tokenizer import OHLCTokenizer, TokenizerConfig, create_sequences
from common.training_monitor import EpochMonitor
from common.distributed_utils import MultiGPUWrapper, get_gpu_memory_info
from crypto_data_loader import CryptoMultiAssetLoader

# Import multi-asset model
from common.market_gpt import MarketGPT, MarketGPTConfig


def load_config(config_path: str) -> dict:
    """Load configuration from JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main(args):
    print("="*80)
    print("CASE C: CRYPTO PROTOTYPE TRAINING")
    print("Multi-Asset Model - 20 Cryptocurrencies")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # GPU info
    gpu_info = get_gpu_memory_info()
    if args.num_gpus == "auto":
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = int(args.num_gpus)

    print(f"\n🖥️  GPU Configuration:")
    print(f"  Available GPUs: {gpu_info['num_gpus']}")
    print(f"  Using GPUs: {num_gpus}")
    for i in range(min(num_gpus, len(gpu_info['devices']))):
        gpu = gpu_info['devices'][i]
        print(f"  GPU {i}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_allocated_gb']:.1f}GB / {gpu['memory_total_gb']:.1f}GB")
    print()

    # Load config
    config = load_config(args.config)

    # Setup directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: LOADING CRYPTO DATA")
    print("="*80 + "\n")

    loader = CryptoMultiAssetLoader(cache_dir=config['data']['cache_dir'])

    # Download all cryptos
    df = loader.download_all_cryptos(
        start_date=config['data']['start_date'],
        end_date=config['data'].get('end_date', None),
        use_cache=config['data']['use_cache']
    )

    # Get stats
    stats = loader.get_data_stats(df)
    print("\nDataset statistics:")
    print(f"  Total candles: {stats['total_candles']:,}")
    print(f"  Assets: {stats['num_assets']}")
    print(f"  Date range: {stats['date_range']['days']} days")

    # Split data
    train_df, val_df, test_df = loader.split_train_val_test(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    # Extract arrays
    train_ohlc = loader.get_ohlc_array(train_df)
    train_asset_ids = loader.get_asset_ids(train_df)

    val_ohlc = loader.get_ohlc_array(val_df)
    val_asset_ids = loader.get_asset_ids(val_df)

    test_ohlc = loader.get_ohlc_array(test_df)
    test_asset_ids = loader.get_asset_ids(test_df)

    print(f"\n📊 Data split:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # Save data info
    data_info = {
        'stats': stats,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'assets': loader.CRYPTO_ASSETS,
        'asset_mapping': loader.asset_to_id
    }

    with open(output_path / "data_info.json", 'w') as f:
        json.dump(data_info, f, indent=2)

    # ========================================
    # STEP 2: Tokenizer
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: TOKENIZER")
    print("="*80 + "\n")

    tokenizer_config = TokenizerConfig(
        vocab_size=config['tokenizer']['vocab_size'],
        n_features=4
    )

    tokenizer = OHLCTokenizer(tokenizer_config)
    tokenizer.fit(train_ohlc)
    tokenizer.save(output_path / "tokenizer.pkl")

    vocab_stats = tokenizer.get_vocab_stats()
    print("Vocabulary statistics:")
    for key, value in vocab_stats.items():
        print(f"  {key}: {value}")

    # Tokenize data
    print("\nTokenizing data...")
    train_tokens = tokenizer.encode(train_ohlc)
    val_tokens = tokenizer.encode(val_ohlc)
    test_tokens = tokenizer.encode(test_ohlc)

    # Create sequences
    print("Creating sequences...")
    seq_length = config['model']['context_length'] // 2
    stride = seq_length // 2

    X_train, y_train = create_sequences(train_tokens, seq_length, stride)
    X_val, y_val = create_sequences(val_tokens, seq_length, stride)
    X_test, y_test = create_sequences(test_tokens, seq_length, stride)

    # Match asset_ids with sequences
    # For simplicity, take first asset_id of each sequence
    train_asset_seq = np.array([
        train_asset_ids[i * stride]
        for i in range(len(X_train))
    ])
    val_asset_seq = np.array([
        val_asset_ids[i * stride]
        for i in range(len(X_val))
    ])
    test_asset_seq = np.array([
        test_asset_ids[i * stride]
        for i in range(len(X_test))
    ])

    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Val sequences: {len(X_val):,}")
    print(f"  Test sequences: {len(X_test):,}")
    print(f"  Sequence length: {seq_length}")

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).long(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(train_asset_seq).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).long(),
        torch.from_numpy(y_val).long(),
        torch.from_numpy(val_asset_seq).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).long(),
        torch.from_numpy(y_test).long(),
        torch.from_numpy(test_asset_seq).long()
    )

    # Adjust batch size for multi-GPU
    base_batch_size = config['training']['batch_size']
    effective_batch_size = base_batch_size * num_gpus

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n🎯 Training configuration:")
    print(f"  Base batch size: {base_batch_size}")
    print(f"  Effective batch size: {effective_batch_size} ({base_batch_size} × {num_gpus} GPUs)")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ========================================
    # STEP 3: Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: MODEL CREATION")
    print("="*80 + "\n")

    model_config = MarketGPTConfig(
        n_layers=config['model']['n_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['context_length'],
        vocab_size=config['tokenizer']['vocab_size'],
        n_features=4,
        n_steps_pred=config['model'].get('n_steps_pred', 5),
        dropout=config['model']['dropout'],
        gradient_checkpointing=args.gradient_checkpointing
    )

    model = MarketGPT(model_config)

    # Wrap for multi-GPU if needed
    if num_gpus > 1:
        print(f"\n🔧 Wrapping model for {num_gpus} GPUs...")
        model = MultiGPUWrapper(model, num_gpus=num_gpus)
    else:
        model = model.to(args.device)

    param_counts = model.count_parameters() if num_gpus == 1 else model.module.count_parameters()
    print(f"Model parameters: {param_counts['total']:,}")

    # ========================================
    # STEP 4: Training
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80 + "\n")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=config['training']['weight_decay']
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Monitor
    monitor = EpochMonitor(
        total_epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader)
    )

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config['training']['epochs']):
        monitor.start_epoch(epoch)

        # Train
        model.train()
        train_loss_sum = 0

        for batch_idx, (token_ids, targets, asset_ids) in enumerate(train_loader):
            token_ids = token_ids.to(args.device)
            targets = targets.to(args.device)
            asset_ids = asset_ids.to(args.device)

            # Forward pass (note: base model doesn't use asset_ids yet)
            # For now, we'll train without asset conditioning
            # TODO: Switch to MarketGPTMultiAsset
            next_logits, _ = model(token_ids, mode="next_token")

            # Compute loss
            B, T = token_ids.shape[:2]
            vocab_size = next_logits.shape[-1] // 4

            # next_logits shape: (B, T, 4*vocab_size)
            # Reshape to (B, T, 4, vocab_size)
            logits_reshaped = next_logits.view(B, T, 4, vocab_size)

            # We only predict the next token after the sequence
            # Take last timestep: (B, 4, vocab_size)
            logits_last = logits_reshaped[:, -1, :, :]

            loss = 0
            for i in range(4):
                loss += criterion(
                    logits_last[:, i, :],  # (B, vocab_size)
                    targets[:, i]           # (B,)
                )
            loss /= 4

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for token_ids, targets, asset_ids in val_loader:
                token_ids = token_ids.to(args.device)
                targets = targets.to(args.device)

                next_logits, _ = model(token_ids, mode="next_token")

                B, T = token_ids.shape[:2]
                vocab_size = next_logits.shape[-1] // 4
                logits_reshaped = next_logits.view(B, T, 4, vocab_size)

                # Take last timestep prediction
                logits_last = logits_reshaped[:, -1, :, :]

                loss = 0
                for i in range(4):
                    loss += criterion(
                        logits_last[:, i, :],  # (B, vocab_size)
                        targets[:, i]           # (B,)
                    )
                loss /= 4

                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)

        # Update monitor
        monitor.end_epoch(train_loss, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_path / "best_model.pt")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / f"checkpoint_epoch_{epoch+1}.pt")

    # ========================================
    # STEP 5: Test Set Evaluation
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")

    # Load best model
    print("Loading best model...")
    checkpoint = torch.load(output_path / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate on test set
    test_loss_sum = 0
    test_correct = 0
    test_total = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for token_ids, targets, asset_ids in test_loader:
            token_ids = token_ids.to(args.device)
            targets = targets.to(args.device)

            next_logits, _ = model(token_ids, mode="next_token")

            B, T = token_ids.shape[:2]
            vocab_size = next_logits.shape[-1] // 4
            logits_reshaped = next_logits.view(B, T, 4, vocab_size)

            # Take last timestep prediction
            logits_last = logits_reshaped[:, -1, :, :]

            loss = 0
            for i in range(4):
                loss += criterion(
                    logits_last[:, i, :],  # (B, vocab_size)
                    targets[:, i]           # (B,)
                )
            loss /= 4

            test_loss_sum += loss.item()

            # Calculate accuracy
            for i in range(4):
                preds = logits_last[:, i, :].argmax(dim=-1)
                test_correct += (preds == targets[:, i]).sum().item()
                test_total += targets[:, i].numel()

    test_loss = test_loss_sum / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    print(f"\n📊 Test Set Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Test Samples: {len(test_dataset):,}")

    # ========================================
    # DONE
    # ========================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Model saved to: {output_path}/best_model.pt")
    print(f"Tokenizer saved to: {output_path}/tokenizer.pkl")
    print("="*80 + "\n")

    # Save final summary
    summary = monitor.get_summary()
    summary['best_val_loss'] = best_val_loss
    summary['test_loss'] = test_loss
    summary['test_accuracy'] = test_accuracy
    summary['test_samples'] = len(test_dataset)
    summary['final_time'] = datetime.now().isoformat()

    with open(output_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crypto Multi-Asset Model")

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--num-gpus', type=str, default='auto',
                       help='Number of GPUs to use (auto/1/2/...)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    main(args)
