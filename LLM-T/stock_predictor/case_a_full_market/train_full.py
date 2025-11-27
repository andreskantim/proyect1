"""
Training script for Case A: Full Market (600 assets)
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
from universal_loader import UniversalMarketDataLoader


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
    print("Case A: Full Market Training (600 assets)")
    print("=" * 60)

    # GPU info
    gpu_info = get_gpu_memory_info()
    if num_gpus == "auto":
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = int(num_gpus)

    print(f"\n🖥️  GPU Configuration:")
    print(f"  Available GPUs: {gpu_info['num_gpus']}")
    print(f"  Using GPUs: {num_gpus}")
    for i in range(min(num_gpus, len(gpu_info['devices']))):
        gpu = gpu_info['devices'][i]
        print(f"  GPU {i}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_allocated_gb']:.1f}GB / {gpu['memory_total_gb']:.1f}GB")

    # Initialize tokenizer
    print("\n📊 Initializing tokenizer...")
    tokenizer = OHLCTokenizer(
        num_bins=config['model']['vocab_size'],
        method='quantile'
    )

    # Load data
    print("\n📥 Loading universal market data...")
    loader = UniversalMarketDataLoader(
        start_date=config['data']['start_date'],
        cache_dir=config['data'].get('cache_dir', 'data/cache_full')
    )

    df = loader.load_all_data()

    if df is None or len(df) == 0:
        print("❌ Failed to load data or no data available")
        return

    print(f"\n📈 Data loaded:")
    print(f"  Total assets: {df['asset_id'].nunique()}")
    print(f"  Total datapoints: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Categories: {df['category_id'].nunique()}")

    # Fit tokenizer on all data
    print("\n🔧 Fitting tokenizer on full dataset...")
    all_ohlc = df[['Open', 'High', 'Low', 'Close']].values
    tokenizer.fit(all_ohlc)

    # Split data
    print("\n✂️  Splitting data...")
    train_size = int(len(df) * config['data']['train_ratio'])
    val_size = int(len(df) * config['data']['val_ratio'])

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # Create sequences
    context_length = config['model']['context_length']

    print("\n🔄 Creating training sequences...")
    train_seqs, train_assets, train_cats = create_sequences(train_df, tokenizer, context_length)
    print(f"  Train sequences: {len(train_seqs):,}")

    print("🔄 Creating validation sequences...")
    val_seqs, val_assets, val_cats = create_sequences(val_df, tokenizer, context_length)
    print(f"  Val sequences: {len(val_seqs):,}")

    print("🔄 Creating test sequences...")
    test_seqs, test_assets, test_cats = create_sequences(test_df, tokenizer, context_length)
    print(f"  Test sequences: {len(test_seqs):,}")

    # Create datasets
    train_dataset = MultiAssetDataset(
        train_seqs[:, :-1],
        train_assets,
        train_cats,
        context_length
    )

    val_dataset = MultiAssetDataset(
        val_seqs[:, :-1],
        val_assets,
        val_cats,
        context_length
    )

    test_dataset = MultiAssetDataset(
        test_seqs[:, :-1],
        test_assets,
        test_cats,
        context_length
    )

    # Create dataloaders - adjust batch size based on number of GPUs
    base_batch_size = config['training']['batch_size']
    effective_batch_size = base_batch_size * num_gpus

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    print(f"\n🎯 Training configuration:")
    print(f"  Base batch size: {base_batch_size}")
    print(f"  Effective batch size: {effective_batch_size} ({base_batch_size} × {num_gpus} GPUs)")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Initialize model
    print("\n🤖 Initializing model...")
    model_config = MultiAssetConfig(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        num_assets=df['asset_id'].nunique(),
        num_categories=df['category_id'].nunique(),
        asset_embed_dim=config['model'].get('asset_embed_dim', 32),
        category_embed_dim=config['model'].get('category_embed_dim', 16)
    )

    model = MarketGPTMultiAsset(model_config)

    # Wrap for multi-GPU if needed
    if num_gpus > 1:
        print(f"\n🔧 Wrapping model for {num_gpus} GPUs...")
        model = MultiGPUWrapper(model, num_gpus=num_gpus)
    else:
        model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        betas=(0.9, 0.95)
    )

    # Cosine annealing with warmup
    warmup_steps = config['training'].get('warmup_steps', 1000)
    total_steps = len(train_loader) * config['training']['epochs']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training monitor
    monitor = EpochMonitor(output_dir)

    # Training loop
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience', 10)
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for batch_idx, (x, y, asset_ids, category_ids) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            asset_ids = asset_ids.squeeze(1).to(device)
            category_ids = category_ids.squeeze(1).to(device)

            optimizer.zero_grad()

            logits = model(x, asset_ids, category_ids)

            # Reshape for loss
            loss = criterion(
                logits.view(-1, config['model']['vocab_size']),
                y.view(-1)
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training'].get('grad_clip', 1.0)
            )

            optimizer.step()
            scheduler.step()

            # Metrics
            train_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            train_correct += (predictions == y).sum().item()
            train_total += y.numel()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*train_correct/train_total:.2f}%",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y, asset_ids, category_ids in tqdm(val_loader, desc="Validation"):
                x = x.to(device)
                y = y.to(device)
                asset_ids = asset_ids.squeeze(1).to(device)
                category_ids = category_ids.squeeze(1).to(device)

                logits = model(x, asset_ids, category_ids)

                loss = criterion(
                    logits.view(-1, config['model']['vocab_size']),
                    y.view(-1)
                )

                val_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                val_correct += (predictions == y).sum().item()
                val_total += y.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Log epoch
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': scheduler.get_last_lr()[0]
        }

        monitor.log_epoch(epoch_stats)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'config': config,
                'num_assets': df['asset_id'].nunique(),
                'num_categories': df['category_id'].nunique()
            }

            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint every N epochs
        if (epoch + 1) % config['training'].get('checkpoint_every', 10) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("📊 Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    print("\n Loading best model...")
    best_checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    # Evaluate on test set
    test_loss = 0
    test_correct = 0
    test_total = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for x, y, asset_ids, category_ids in tqdm(test_loader, desc="Testing"):
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

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    # Save final artifacts
    print("\n💾 Saving final artifacts...")

    # Save tokenizer
    with open(output_dir / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save asset mapping
    asset_info = {
        'asset_to_id': loader.asset_to_id,
        'id_to_asset': loader.id_to_asset,
        'category_to_id': loader.category_to_id,
        'id_to_category': loader.id_to_category,
        'num_assets': df['asset_id'].nunique(),
        'num_categories': df['category_id'].nunique()
    }

    with open(output_dir / 'asset_info.json', 'w') as f:
        json.dump(asset_info, f, indent=2)

    print("\n✅ Training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final test loss: {avg_test_loss:.4f}")
    print(f"  Final test accuracy: {test_accuracy:.2f}%")
    print(f"📁 Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Case A: Full Market (600 assets)")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-gpus", type=str, default="auto", help="Number of GPUs (auto/1/2/...)")

    args = parser.parse_args()

    train(args.config, args.output, args.device, args.num_gpus)
