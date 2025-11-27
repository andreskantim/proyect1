#!/bin/bash
#SBATCH --job-name=diag_t4             # Job name
#SBATCH --partition=viz                # Partition with T4 GPUs
#SBATCH --nodes=1                      # Nodes
#SBATCH --ntasks=1                     # Tasks
#SBATCH --cpus-per-task=16             # CPUs
#SBATCH --gres=gpu:t4:1                # 1 T4 GPU
#SBATCH --mem=48G                      # Memory
#SBATCH --time=6:00:00                 # 6 hours limit
#SBATCH --output=../logs/diagnostic_t4_%j.out
#SBATCH --error=../logs/diagnostic_t4_%j.err

cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/case_c_crypto

echo "================================================"
echo "Case C: DIAGNOSTIC TEST on T4 GPU"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

# Activate environment
source ~/.bashrc
conda activate llm-training

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p ../checkpoints/case_c_crypto_diagnostic

# GPU info
echo ""
nvidia-smi
echo ""

# Run diagnostic training with enhanced metrics
echo "Starting diagnostic training..."

python -u << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor')

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Import modules
from case_c_crypto.crypto_data_loader import CryptoMultiAssetLoader
from case_c_crypto.diagnostic_metrics import (
    compute_feature_accuracy,
    analyze_prediction_distribution,
    compute_per_feature_loss,
    print_diagnostic_summary,
    is_model_learning
)
from common.tokenizer import OHLCTokenizer, TokenizerConfig, create_sequences
from common.market_gpt import MarketGPT
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("DIAGNOSTIC TRAINING - Case C Crypto")
print("=" * 80)

# Load config
with open('configs/crypto_t4_diagnostic.json') as f:
    config = json.load(f)

print(f"\n📋 Configuration:")
print(f"  Epochs: {config['training']['epochs']}")
print(f"  Batch size: {config['training']['batch_size']}")
print(f"  Model: {config['model']['n_layers']}L, {config['model']['d_model']}D")
print(f"  Vocab size: {config['tokenizer']['vocab_size']}")
print(f"  Context length: {config['model']['context_length']}")

# Load data
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
loader = CryptoMultiAssetLoader()
df = loader.download_crypto_data(
    start_date=config['data']['start_date'],
    end_date=config['data']['end_date'],
    cache_dir=config['data']['cache_dir'],
    use_cache=config['data']['use_cache']
)

train_df, val_df, test_df = loader.split_train_val_test(
    df,
    train_ratio=config['data']['train_ratio'],
    val_ratio=config['data']['val_ratio']
)

train_ohlc = loader.get_ohlc_array(train_df)
val_ohlc = loader.get_ohlc_array(val_df)
test_ohlc = loader.get_ohlc_array(test_df)

print(f"\n📊 Dataset sizes:")
print(f"  Train: {len(train_ohlc):,} samples")
print(f"  Val:   {len(val_ohlc):,} samples")
print(f"  Test:  {len(test_ohlc):,} samples")

# Tokenize
print("\n" + "=" * 80)
print("TOKENIZING")
print("=" * 80)
tokenizer = OHLCTokenizer(TokenizerConfig(
    vocab_size=config['tokenizer']['vocab_size'],
    n_features=config['tokenizer']['n_features']
))
tokenizer.fit(train_ohlc)

train_tokens = tokenizer.transform(train_ohlc)
val_tokens = tokenizer.transform(val_ohlc)
test_tokens = tokenizer.transform(test_ohlc)

# Create sequences
seq_length = config['model']['context_length'] // 2
stride = 1

X_train, y_train = create_sequences(train_tokens, seq_length, stride)
X_val, y_val = create_sequences(val_tokens, seq_length, stride)
X_test, y_test = create_sequences(test_tokens, seq_length, stride)

print(f"\n📦 Sequences created:")
print(f"  Train: {len(X_train):,} sequences")
print(f"  Val:   {len(X_val):,} sequences")
print(f"  Test:  {len(X_test):,} sequences")

# Create dataloaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train).long(),
    torch.from_numpy(y_train).long()
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val).long(),
    torch.from_numpy(y_val).long()
)

batch_size = config['training']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model
print("\n" + "=" * 80)
print("MODEL CREATION")
print("=" * 80)
model = MarketGPT(
    vocab_size=config['tokenizer']['vocab_size'],
    n_features=config['tokenizer']['n_features'],
    d_model=config['model']['d_model'],
    n_heads=config['model']['n_heads'],
    n_layers=config['model']['n_layers'],
    d_ff=config['model']['d_ff'],
    context_length=config['model']['context_length'],
    n_steps_pred=config['model']['n_steps_pred'],
    dropout=config['model']['dropout'],
    attention_dropout=config['model']['attention_dropout']
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model created with {n_params:,} parameters")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'feature_accuracy': {f: [] for f in ['Open', 'High', 'Low', 'Close']}
}

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float('inf')
vocab_size = config['tokenizer']['vocab_size']

for epoch in range(1, config['training']['epochs'] + 1):
    # Training
    model.train()
    train_loss_sum = 0

    for token_ids, targets in train_loader:
        token_ids = token_ids.to(device)
        targets = targets.to(device)

        next_logits, _ = model(token_ids, mode="next_token")
        B, T = token_ids.shape[:2]

        logits_reshaped = next_logits.view(B, T, 4, vocab_size)
        logits_last = logits_reshaped[:, -1, :, :]

        loss = 0
        for i in range(4):
            loss += criterion(logits_last[:, i, :], targets[:, i])
        loss /= 4

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        optimizer.step()

        train_loss_sum += loss.item()

    train_loss = train_loss_sum / len(train_loader)

    # Validation
    model.eval()
    val_loss_sum = 0
    val_correct = 0
    val_total = 0

    feature_accs = []
    feature_losses = []
    pred_dists = []

    with torch.no_grad():
        for token_ids, targets in val_loader:
            token_ids = token_ids.to(device)
            targets = targets.to(device)

            next_logits, _ = model(token_ids, mode="next_token")
            B, T = token_ids.shape[:2]

            logits_reshaped = next_logits.view(B, T, 4, vocab_size)
            logits_last = logits_reshaped[:, -1, :, :]

            loss = 0
            for i in range(4):
                loss += criterion(logits_last[:, i, :], targets[:, i])
            loss /= 4

            val_loss_sum += loss.item()

            # Compute detailed metrics
            feature_acc = compute_feature_accuracy(logits_last, targets)
            feature_accs.append(feature_acc)

            feature_loss = compute_per_feature_loss(logits_last, targets, criterion)
            feature_losses.append(feature_loss)

            pred_dist = analyze_prediction_distribution(logits_last, targets, vocab_size)
            pred_dists.append(pred_dist)

            # Overall accuracy
            for i in range(4):
                preds = logits_last[:, i, :].argmax(dim=-1)
                val_correct += (preds == targets[:, i]).sum().item()
                val_total += targets[:, i].numel()

    val_loss = val_loss_sum / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    # Average feature metrics
    avg_feature_acc = {}
    avg_feature_loss = {}
    avg_pred_dist = {}

    for feature in ['Open', 'High', 'Low', 'Close']:
        avg_feature_acc[feature] = np.mean([fa[feature] for fa in feature_accs])
        avg_feature_loss[feature] = np.mean([fl[feature] for fl in feature_losses])

        # Average prediction distribution
        avg_pred_dist[feature] = {
            'unique_predictions': int(np.mean([pd[feature]['unique_predictions'] for pd in pred_dists])),
            'total_vocab_used_pct': np.mean([pd[feature]['total_vocab_used_pct'] for pd in pred_dists]),
            'most_common_token': pred_dists[-1][feature]['most_common_token'],  # Last batch
            'most_common_pct': np.mean([pd[feature]['most_common_pct'] for pd in pred_dists])
        }

    # Save to history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    for feature in ['Open', 'High', 'Low', 'Close']:
        history['feature_accuracy'][feature].append(avg_feature_acc[feature])

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, '../checkpoints/case_c_crypto_diagnostic/best_model.pt')

    # Print diagnostic summary every epoch
    train_metrics = {'loss': train_loss}
    val_metrics = {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'feature_accuracy': avg_feature_acc,
        'feature_loss': avg_feature_loss,
        'pred_distribution': avg_pred_dist
    }

    print_diagnostic_summary(epoch, train_metrics, val_metrics)

    # Check if learning
    is_learning, msg = is_model_learning(history['val_loss'])
    print(f"🧠 Learning status: {msg}")

# Final summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")

# Plot learning curves
print("\n📈 Learning Curves:")
print(f"  Epoch | Train Loss | Val Loss | Val Acc")
print(f"  ------|------------|----------|--------")
for i in range(len(history['train_loss'])):
    print(f"  {i+1:5d} | {history['train_loss'][i]:10.4f} | {history['val_loss'][i]:8.4f} | {history['val_accuracy'][i]:6.2f}%")

print("\n" + "=" * 80)

PYTHON_SCRIPT

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Diagnostic training finished: exit code $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
