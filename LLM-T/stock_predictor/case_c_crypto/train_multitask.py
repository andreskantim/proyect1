"""
Training script for multi-task crypto prediction.

Predicts:
1. Direction of next candle (DOWN/FLAT/UP)
2. Magnitude: whether price exceeds 2σ Bollinger in 4 horizons (1w, 2w, 1m, 2m)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from crypto_data_loader import CryptoMultiAssetLoader
from common.multitask_tokenizer import MultiTaskTokenizer, create_multitask_sequences
from common.multitask_model import MultiTaskTransformer, MultiTaskLoss, compute_metrics
from common.distributed_utils import setup_distributed, cleanup_distributed


class MultiTaskDataset(Dataset):
    """Dataset for multi-task prediction."""

    def __init__(self, X: np.ndarray, y: dict):
        """
        Args:
            X: (n_sequences, seq_len, 4) OHLC sequences
            y: Dictionary with 'direction' and 'magnitude' targets
        """
        self.X = torch.from_numpy(X).float()
        self.y_direction = torch.from_numpy(y['direction']).long()
        self.y_magnitude = torch.from_numpy(y['magnitude']).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'input': self.X[idx],
            'direction': self.y_direction[idx],
            'magnitude': self.y_magnitude[idx]
        }


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_dir_loss = 0
    total_mag_loss = 0
    all_predictions = {'direction': [], 'magnitude': []}
    all_targets = {'direction': [], 'magnitude': []}

    for batch in dataloader:
        # Move to device
        inputs = batch['input'].to(device)
        targets = {
            'direction': batch['direction'].to(device),
            'magnitude': batch['magnitude'].to(device)
        }

        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs)

        # Compute loss
        loss, loss_dict = criterion(predictions, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict['total']
        total_dir_loss += loss_dict['direction']
        total_mag_loss += loss_dict['magnitude']

        # Store predictions and targets for metrics
        all_predictions['direction'].append(predictions['direction'].detach())
        all_predictions['magnitude'].append(predictions['magnitude'].detach())
        all_targets['direction'].append(targets['direction'].detach())
        all_targets['magnitude'].append(targets['magnitude'].detach())

    # Compute average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_dir_loss = total_dir_loss / n_batches
    avg_mag_loss = total_mag_loss / n_batches

    # Compute metrics
    all_predictions['direction'] = torch.cat(all_predictions['direction'])
    all_predictions['magnitude'] = torch.cat(all_predictions['magnitude'])
    all_targets['direction'] = torch.cat(all_targets['direction'])
    all_targets['magnitude'] = torch.cat(all_targets['magnitude'])

    metrics = compute_metrics(all_predictions, all_targets)

    return {
        'loss': avg_loss,
        'dir_loss': avg_dir_loss,
        'mag_loss': avg_mag_loss,
        **metrics
    }


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0
    total_dir_loss = 0
    total_mag_loss = 0
    all_predictions = {'direction': [], 'magnitude': []}
    all_targets = {'direction': [], 'magnitude': []}

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            inputs = batch['input'].to(device)
            targets = {
                'direction': batch['direction'].to(device),
                'magnitude': batch['magnitude'].to(device)
            }

            # Forward pass
            predictions = model(inputs)

            # Compute loss
            loss, loss_dict = criterion(predictions, targets)

            # Accumulate metrics
            total_loss += loss_dict['total']
            total_dir_loss += loss_dict['direction']
            total_mag_loss += loss_dict['magnitude']

            # Store predictions and targets
            all_predictions['direction'].append(predictions['direction'])
            all_predictions['magnitude'].append(predictions['magnitude'])
            all_targets['direction'].append(targets['direction'])
            all_targets['magnitude'].append(targets['magnitude'])

    # Compute average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_dir_loss = total_dir_loss / n_batches
    avg_mag_loss = total_mag_loss / n_batches

    # Compute metrics
    all_predictions['direction'] = torch.cat(all_predictions['direction'])
    all_predictions['magnitude'] = torch.cat(all_predictions['magnitude'])
    all_targets['direction'] = torch.cat(all_targets['direction'])
    all_targets['magnitude'] = torch.cat(all_targets['magnitude'])

    metrics = compute_metrics(all_predictions, all_targets)

    return {
        'loss': avg_loss,
        'dir_loss': avg_dir_loss,
        'mag_loss': avg_mag_loss,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    print("="*80)
    print("CASE C: CRYPTO MULTI-TASK TRAINING")
    print("Direction + Magnitude Prediction (2σ Bollinger)")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  GPU Configuration:")
    if torch.cuda.is_available():
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("="*80)
    print("STEP 1: LOADING CRYPTO DATA")
    print("="*80)
    print()

    data_cfg = config['data']
    loader = CryptoMultiAssetLoader(
        cache_dir=data_cfg['cache_dir']
    )

    df = loader.download_all_cryptos(
        start_date=data_cfg['start_date'],
        end_date=data_cfg.get('end_date'),
        use_cache=data_cfg['use_cache']
    )

    print(f"\nDataset statistics:")
    print(f"  Total candles: {len(df):,}")
    print(f"  Assets: {df['asset_id'].nunique()}")
    print(f"  Date range: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # Split data
    train_size = int(len(df) * data_cfg['train_ratio'])
    val_size = int(len(df) * data_cfg['val_ratio'])

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"Data split:")
    print(f"  Train: {len(train_df):,} candles")
    print(f"  Val:   {len(val_df):,} candles")
    print(f"  Test:  {len(test_df):,} candles")

    # Extract OHLC
    train_ohlc = train_df[['open', 'high', 'low', 'close']].values
    val_ohlc = val_df[['open', 'high', 'low', 'close']].values
    test_ohlc = test_df[['open', 'high', 'low', 'close']].values

    # =========================================================================
    # STEP 2: TOKENIZER
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: MULTI-TASK TOKENIZER")
    print("="*80)
    print()

    tokenizer_cfg = config['tokenizer']
    tokenizer = MultiTaskTokenizer(
        direction_threshold=tokenizer_cfg['direction_threshold'],
        bollinger_window=tokenizer_cfg['bollinger_window'],
        bollinger_std=tokenizer_cfg['bollinger_std'],
        horizons=tokenizer_cfg['horizons']
    )

    tokenizer.fit(train_ohlc)

    # Transform data
    print("\nTokenizing data...")
    train_targets = tokenizer.transform(train_ohlc)
    val_targets = tokenizer.transform(val_ohlc)
    test_targets = tokenizer.transform(test_ohlc)

    # Print statistics
    print("\nTraining data statistics:")
    train_stats = tokenizer.get_stats(train_targets)
    print(f"  Direction:")
    for cls, pct in [(k, v) for k, v in train_stats['direction'].items() if 'pct' in k]:
        print(f"    {cls.replace('_pct', '')}: {pct:.1f}%")
    print(f"  Magnitude (2σ Bollinger):")
    for horizon, stats in train_stats['magnitude'].items():
        print(f"    {horizon}: {stats['significant_pct']:.1f}% significant")

    # Create sequences
    print("\nCreating sequences...")
    context_length = config['model']['context_length']
    train_X, train_y = create_multitask_sequences(train_ohlc, train_targets, context_length, stride=1)
    val_X, val_y = create_multitask_sequences(val_ohlc, val_targets, context_length, stride=1)
    test_X, test_y = create_multitask_sequences(test_ohlc, test_targets, context_length, stride=1)

    print(f"  Train sequences: {len(train_X):,}")
    print(f"  Val sequences:   {len(val_X):,}")
    print(f"  Test sequences:  {len(test_X):,}")
    print(f"  Sequence length: {context_length}")

    # Save tokenizer
    tokenizer_path = output_dir / 'tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"\nTokenizer saved to {tokenizer_path}")

    # Create datasets and dataloaders
    batch_size = config['training']['batch_size']
    train_dataset = MultiTaskDataset(train_X, train_y)
    val_dataset = MultiTaskDataset(val_X, val_y)
    test_dataset = MultiTaskDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n🎯 Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # STEP 3: MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL CREATION")
    print("="*80)
    print()

    model_cfg = config['model']
    model = MultiTaskTransformer(
        n_features=4,
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'],
        context_length=model_cfg['context_length'],
        n_direction_classes=model_cfg['n_direction_classes'],
        n_magnitude_tasks=model_cfg['n_magnitude_tasks'],
        dropout=model_cfg['dropout'],
        attention_dropout=model_cfg.get('attention_dropout', 0.1)
    )

    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"Sequences per parameter: {len(train_X) / n_params:.2f}")

    # =========================================================================
    # STEP 4: TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    print()

    train_cfg = config['training']
    task_weights = train_cfg.get('task_weights', {'direction': 1.0, 'magnitude': 1.0})

    criterion = MultiTaskLoss(
        w_direction=task_weights['direction'],
        w_magnitude=task_weights['magnitude']
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )

    # Training loop
    best_val_loss = float('inf')
    patience = train_cfg.get('early_stopping_patience', 10)
    patience_counter = 0

    for epoch in range(1, train_cfg['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{train_cfg['epochs']}")
        print('='*70)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, train_cfg['grad_clip'])

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Print results
        print(f"\nEpoch {epoch} complete:")
        print(f"  Train loss: {train_metrics['loss']:.4f} (dir: {train_metrics['dir_loss']:.4f}, mag: {train_metrics['mag_loss']:.4f})")
        print(f"  Val loss:   {val_metrics['loss']:.4f} (dir: {val_metrics['dir_loss']:.4f}, mag: {val_metrics['mag_loss']:.4f})")
        print(f"\n  Train accuracies:")
        print(f"    Direction: {train_metrics['direction_acc']:.2f}%")
        print(f"    Magnitude: {train_metrics['magnitude_avg_acc']:.2f}% (1w: {train_metrics['magnitude_1w_acc']:.1f}%, 2w: {train_metrics['magnitude_2w_acc']:.1f}%, 1m: {train_metrics['magnitude_1m_acc']:.1f}%, 2m: {train_metrics['magnitude_2m_acc']:.1f}%)")
        print(f"\n  Val accuracies:")
        print(f"    Direction: {val_metrics['direction_acc']:.2f}%")
        print(f"    Magnitude: {val_metrics['magnitude_avg_acc']:.2f}% (1w: {val_metrics['magnitude_1w_acc']:.1f}%, 2w: {val_metrics['magnitude_2w_acc']:.1f}%, 1m: {val_metrics['magnitude_1m_acc']:.1f}%, 2m: {val_metrics['magnitude_2m_acc']:.1f}%)")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            print(f"  ✓ BEST - Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n  Early stopping triggered!")
                break

    # =========================================================================
    # STEP 5: FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: FINAL EVALUATION ON TEST SET")
    print("="*80)
    print()

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    test_metrics = validate_epoch(model, test_loader, criterion, device)

    print(f"📊 Test Set Results:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']:.2f}%")
    print(f"  Magnitude Accuracies:")
    print(f"    1w: {test_metrics['magnitude_1w_acc']:.2f}%")
    print(f"    2w: {test_metrics['magnitude_2w_acc']:.2f}%")
    print(f"    1m: {test_metrics['magnitude_1m_acc']:.2f}%")
    print(f"    2m: {test_metrics['magnitude_2m_acc']:.2f}%")
    print(f"    Average: {test_metrics['magnitude_avg_acc']:.2f}%")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test loss: {test_metrics['loss']:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print(f"Tokenizer saved to: {output_dir / 'tokenizer.pkl'}")
    print("="*80)


if __name__ == "__main__":
    main()
