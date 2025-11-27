"""
Training script for Case B: Reduced (100 assets)

Supports multi-GPU training via DataParallel.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

# Add common modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from market_gpt_multi import MarketGPTMultiAsset, MultiAssetConfig
from tokenizer import OHLCTokenizer
from training_monitor import EpochMonitor
from distributed_utils import MultiGPUWrapper, get_gpu_memory_info
from multi_market_loader import MultiMarketDataLoader


class MultiAssetDataset(Dataset):
    """Dataset for multi-asset training with asset IDs."""

    def __init__(self, token_ids, asset_ids, category_ids, context_length):
        self.token_ids = token_ids
        self.asset_ids = asset_ids
        self.category_ids = category_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.context_length]
        y = self.token_ids[idx+1:idx+self.context_length+1]
        asset_id = self.asset_ids[idx]
        category_id = self.category_ids[idx]

        return (
            torch.LongTensor(x),
            torch.LongTensor(y),
            torch.LongTensor([asset_id]),
            torch.LongTensor([category_id])
        )


def create_sequences(df, tokenizer, context_length):
    """Create training sequences from dataframe."""
    sequences = []
    asset_ids = []
    category_ids = []

    # Group by asset to maintain temporal continuity
    for asset_id in df['asset_id'].unique():
        asset_df = df[df['asset_id'] == asset_id].sort_index()

        if len(asset_df) < context_length + 10:
            continue

        # Extract OHLC
        ohlc = asset_df[['Open', 'High', 'Low', 'Close']].values

        # Tokenize
        tokens = tokenizer.encode(ohlc)

        # Create sequences
        for i in range(len(tokens) - context_length):
            sequences.append(tokens[i:i+context_length+1])
            asset_ids.append(asset_id)
            category_ids.append(asset_df['category_id'].iloc[0])

    return np.array(sequences), np.array(asset_ids), np.array(category_ids)


def train(config_path, output_dir, device="cuda", num_gpus="auto"):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Case B: Reduced Multi-Market Training (100 assets)")
    print("=" * 60)

    # GPU info
    print(f"\nGPU Setup:")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(get_gpu_memory_info())

    # Load data
    print("\n" + "=" * 60)
    print("1. Loading Data")
    print("=" * 60)

    loader = MultiMarketDataLoader()
    all_data = loader.download_all_assets(
        start_date=config['data']['start_date'],
        force_redownload=False
    )

    train_df, val_df, test_df = loader.prepare_training_data(
        all_data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    # Save data info
    data_info = {
        'num_assets': len(all_data),
        'total_candles': len(train_df) + len(val_df) + len(test_df),
        'train_candles': len(train_df),
        'val_candles': len(val_df),
        'test_candles': len(test_df),
        'date_range': {
            'start': str(train_df.index.min()),
            'end': str(test_df.index.max())
        },
        'categories': loader.CATEGORIES
    }

    with open(output_dir / "data_info.json", "w") as f:
        json.dump(data_info, f, indent=2)

    print(f"\nCategory distribution (train):")
    print(loader.get_category_stats(train_df))

    # Fit tokenizer
    print("\n" + "=" * 60)
    print("2. Fitting Tokenizer")
    print("=" * 60)

    tokenizer = OHLCTokenizer(vocab_size=config['model']['vocab_size'])
    train_ohlc = train_df[['Open', 'High', 'Low', 'Close']].values
    tokenizer.fit(train_ohlc)

    # Save tokenizer
    with open(output_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Tokenizer fitted with vocab_size={config['model']['vocab_size']}")

    # Create sequences
    print("\n" + "=" * 60)
    print("3. Creating Sequences")
    print("=" * 60)

    context_length = config['model']['context_length']

    print("Creating training sequences...")
    train_seq, train_assets, train_cats = create_sequences(train_df, tokenizer, context_length)

    print("Creating validation sequences...")
    val_seq, val_assets, val_cats = create_sequences(val_df, tokenizer, context_length)

    print("Creating test sequences...")
    test_seq, test_assets, test_cats = create_sequences(test_df, tokenizer, context_length)

    print(f"\nSequences created:")
    print(f"  Train: {len(train_seq):,} sequences")
    print(f"  Val:   {len(val_seq):,} sequences")
    print(f"  Test:  {len(test_seq):,} sequences")

    # Create datasets and dataloaders
    train_dataset = MultiAssetDataset(train_seq, train_assets, train_cats, context_length)
    val_dataset = MultiAssetDataset(val_seq, val_assets, val_cats, context_length)
    test_dataset = MultiAssetDataset(test_seq, test_assets, test_cats, context_length)

    # Batch size adjustment for multi-GPU
    base_batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=base_batch_size,  # Will be adjusted by MultiGPUWrapper
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=base_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=base_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create model
    print("\n" + "=" * 60)
    print("4. Creating Model")
    print("=" * 60)

    model_config = MultiAssetConfig(
        vocab_size=config['model']['vocab_size'],
        n_layers=config['model']['n_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        context_length=context_length,
        dropout=config['model'].get('dropout', 0.1),
        num_assets=len(loader.all_assets),
        num_categories=len(loader.CATEGORIES)
    )

    model = MarketGPTMultiAsset(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Layers: {config['model']['n_layers']}")
    print(f"  Model dim: {config['model']['d_model']}")
    print(f"  Heads: {config['model']['n_heads']}")

    # Wrap with multi-GPU support
    gpu_wrapper = MultiGPUWrapper(model, device=device, strategy="auto")
    model = gpu_wrapper.get_wrapped_model()

    # Adjust batch size if using DataParallel
    actual_batch_size = gpu_wrapper.adjust_batch_size(base_batch_size)
    if actual_batch_size != base_batch_size:
        print(f"\nBatch size adjusted for {torch.cuda.device_count()} GPUs:")
        print(f"  Base batch size: {base_batch_size}")
        print(f"  Actual batch size: {actual_batch_size}")

        # Recreate dataloaders with adjusted batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training'].get('scheduler_t0', 5),
        T_mult=2
    )

    # Training loop
    print("\n" + "=" * 60)
    print("5. Training")
    print("=" * 60)

    best_val_loss = float('inf')
    epochs = config['training']['epochs']

    epoch_monitor = EpochMonitor(total_epochs=epochs)

    for epoch in range(epochs):
        epoch_monitor.start_epoch()

        # Training
        model.train()
        train_loss = 0.0

        for token_ids, targets, asset_ids, category_ids in train_loader:
            token_ids = token_ids.to(device)
            targets = targets.to(device)
            asset_ids = asset_ids.squeeze(1).to(device)
            category_ids = category_ids.squeeze(1).to(device)

            optimizer.zero_grad()

            # Forward pass
            next_logits, _ = model(token_ids, asset_ids, category_ids, mode="next_token")

            # Compute loss for each OHLC feature
            batch_size, seq_len, n_features, vocab_size = next_logits.shape
            loss = sum([
                criterion(
                    next_logits[:, :, i, :].reshape(-1, vocab_size),
                    targets[:, :, i].reshape(-1)
                ) for i in range(n_features)
            ]) / n_features

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for token_ids, targets, asset_ids, category_ids in val_loader:
                token_ids = token_ids.to(device)
                targets = targets.to(device)
                asset_ids = asset_ids.squeeze(1).to(device)
                category_ids = category_ids.squeeze(1).to(device)

                next_logits, _ = model(token_ids, asset_ids, category_ids, mode="next_token")

                batch_size, seq_len, n_features, vocab_size = next_logits.shape
                loss = sum([
                    criterion(
                        next_logits[:, :, i, :].reshape(-1, vocab_size),
                        targets[:, :, i].reshape(-1)
                    ) for i in range(n_features)
                ]) / n_features

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            gpu_wrapper.save_checkpoint(
                output_dir / "best_model.pt",
                epoch=epoch,
                val_loss=best_val_loss,
                config=config
            )

        # Update monitor
        epoch_monitor.update(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            lr=optimizer.param_groups[0]['lr']
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            gpu_wrapper.save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch+1}.pt",
                epoch=epoch,
                val_loss=avg_val_loss,
                config=config
            )

    epoch_monitor.finish()

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("📊 Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    print("\nLoading best model...")
    gpu_wrapper.load_checkpoint(output_dir / "best_model.pt")
    model.eval()

    # Evaluate on test set
    test_loss = 0
    test_correct = 0
    test_total = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for x, y, asset_ids, category_ids in test_loader:
            x = x.to(device)
            y = y.to(device)
            asset_ids = asset_ids.squeeze(1).to(device)
            category_ids = category_ids.squeeze(1).to(device)

            logits = model(x, asset_ids, category_ids)

            loss = criterion(
                logits.view(-1, config['model']['vocab_size']),
                y.view(-1)
            )

            test_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            test_correct += (predictions == y).sum().item()
            test_total += y.numel()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    print(f"\n📊 Test Set Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Test Samples: {len(test_dataset):,}")

    # Save test results
    test_results = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'test_samples': len(test_dataset),
        'best_val_loss': best_val_loss
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Save training summary
    summary = {
        'total_epochs': epochs,
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'model_params': total_params,
        'num_gpus_used': torch.cuda.device_count() if device == "cuda" else 0
    }

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test loss: {avg_test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    train(args.config, args.output_dir, args.device)
