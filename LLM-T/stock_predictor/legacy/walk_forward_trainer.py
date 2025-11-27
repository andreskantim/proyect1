"""
Walk-Forward Training System for Market GPT.
Implements continuous learning with sequential parameter updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from market_gpt import MarketGPT, MarketGPTConfig
from tokenizer import OHLCTokenizer, TokenizerConfig, create_sequences


class WalkForwardTrainer:
    """
    Walk-forward trainer for Market GPT.

    Implements a rolling window training strategy:
    1. Initial training on historical data
    2. Walk forward: test on next window, then fine-tune on that window
    3. Repeat until all data is processed

    This simulates real-world trading where model is continuously updated
    with new market data.
    """

    def __init__(
        self,
        model: MarketGPT,
        tokenizer: OHLCTokenizer,
        config: Dict,
        device: Optional[str] = None
    ):
        """
        Initialize walk-forward trainer.

        Args:
            model: MarketGPT model
            tokenizer: OHLC tokenizer
            config: Training configuration dict
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizers and schedulers
        self.optimizer = None
        self.scheduler = None

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # History
        self.train_history = []
        self.walk_forward_history = []

        print(f"Walk-forward trainer initialized on {self.device}")

    def _setup_optimizer(self, learning_rate: float, weight_decay: float = 0.1):
        """Setup optimizer and learning rate scheduler."""
        # AdamW optimizer (as used in GPT)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )

        print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")

    def _setup_scheduler(self, warmup_steps: int, max_steps: int):
        """Setup learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        next_token_logits: torch.Tensor,
        multi_step_preds: torch.Tensor,
        targets_next: torch.Tensor,
        targets_multi: torch.Tensor,
        alpha: float = 0.7
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss for next-token and multi-step prediction.

        Args:
            token_ids: Input token IDs (B, T, 4)
            next_token_logits: Next-token predictions (B, T, vocab_size * 4)
            multi_step_preds: Multi-step predictions (B, T, n_steps * 4)
            targets_next: Next-token targets (B, T, 4)
            targets_multi: Multi-step targets (B, T, n_steps, 4)
            alpha: Weight for next-token loss (1-alpha for multi-step)

        Returns:
            - total_loss: Combined loss
            - loss_dict: Dictionary with individual losses
        """
        B, T, _ = token_ids.shape

        # Next-token loss (cross-entropy)
        # Reshape logits to (B, T, 4, vocab_size)
        vocab_size = next_token_logits.shape[-1] // 4
        logits_reshaped = next_token_logits.view(B, T, 4, vocab_size)

        # Compute loss for each feature
        loss_next_token = 0
        for i in range(4):
            loss_next_token += self.ce_loss(
                logits_reshaped[:, :, i, :].reshape(-1, vocab_size),
                targets_next[:, :, i].reshape(-1)
            )
        loss_next_token /= 4

        # Multi-step loss (MSE on log returns)
        # First, decode predictions and targets to get returns
        n_steps = multi_step_preds.shape[-1] // 4
        preds_reshaped = multi_step_preds.view(B, T, n_steps, 4)
        loss_multi_step = self.mse_loss(preds_reshaped, targets_multi)

        # Combined loss
        total_loss = alpha * loss_next_token + (1 - alpha) * loss_multi_step

        loss_dict = {
            'total': total_loss.item(),
            'next_token': loss_next_token.item(),
            'multi_step': loss_multi_step.item()
        }

        return total_loss, loss_dict

    def train_epoch(
        self,
        train_loader: DataLoader,
        alpha: float = 0.7,
        grad_clip: float = 1.0
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with training data
            alpha: Weight for next-token loss
            grad_clip: Gradient clipping threshold

        Returns:
            Dictionary with average losses
        """
        self.model.train()

        total_losses = {'total': 0, 'next_token': 0, 'multi_step': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Unpack batch
            token_ids = batch[0].to(self.device)  # (B, T, 4)
            targets_next = batch[1].to(self.device)  # (B, T, 4)
            targets_multi = batch[2].to(self.device) if len(batch) > 2 else None  # (B, T, n_steps, 4)

            # Forward pass
            next_token_logits, multi_step_preds = self.model(token_ids, mode="both")

            # Compute loss
            if targets_multi is None:
                # Create dummy multi-step targets if not provided
                targets_multi = torch.zeros(
                    (token_ids.shape[0], token_ids.shape[1],
                     self.model.config.n_steps_pred, 4),
                    device=self.device
                )

            loss, loss_dict = self.compute_loss(
                token_ids, next_token_logits, multi_step_preds,
                targets_next, targets_multi, alpha
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Optimizer step
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update statistics
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def validate(self, val_loader: DataLoader, alpha: float = 0.7) -> Dict:
        """
        Validate model.

        Args:
            val_loader: DataLoader with validation data
            alpha: Weight for next-token loss

        Returns:
            Dictionary with average losses
        """
        self.model.eval()

        total_losses = {'total': 0, 'next_token': 0, 'multi_step': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch[0].to(self.device)
                targets_next = batch[1].to(self.device)
                targets_multi = batch[2].to(self.device) if len(batch) > 2 else None

                # Forward pass
                next_token_logits, multi_step_preds = self.model(token_ids, mode="both")

                # Compute loss
                if targets_multi is None:
                    targets_multi = torch.zeros(
                        (token_ids.shape[0], token_ids.shape[1],
                         self.model.config.n_steps_pred, 4),
                        device=self.device
                    )

                _, loss_dict = self.compute_loss(
                    token_ids, next_token_logits, multi_step_preds,
                    targets_next, targets_multi, alpha
                )

                # Update statistics
                for key in total_losses:
                    total_losses[key] += loss_dict[key]
                num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def initial_training(
        self,
        train_data: np.ndarray,
        val_split: float = 0.1,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 6e-4,
        warmup_steps: int = 2000,
        save_dir: str = "checkpoints"
    ) -> Dict:
        """
        Initial training phase on historical data.

        Args:
            train_data: OHLC data for initial training (n_samples, 4)
            val_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for LR scheduler
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("INITIAL TRAINING PHASE")
        print("="*60)

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Tokenize data
        print("\nTokenizing data...")
        token_ids = self.tokenizer.encode(train_data)

        # Create sequences
        print("Creating sequences...")
        sequence_length = self.model.config.context_length // 2  # Use half context for initial training
        X, y_next = create_sequences(token_ids, sequence_length, stride=1)

        print(f"  Sequences: {X.shape}")
        print(f"  Sequence length: {sequence_length}")

        # Split train/val
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_next[:split_idx], y_next[split_idx:]

        print(f"  Train samples: {len(X_train):,}")
        print(f"  Val samples: {len(X_val):,}")

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).long(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).long(),
            torch.from_numpy(y_val).long()
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and scheduler
        max_steps = len(train_loader) * epochs
        self._setup_optimizer(learning_rate)
        self._setup_scheduler(warmup_steps, max_steps)

        print(f"\nTraining configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max steps: {max_steps}")
        print(f"  Warmup steps: {warmup_steps}")

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader)

            # Validate
            val_losses = self.validate(val_loader)

            # Log
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"NextTok: {train_losses['next_token']:.4f}, "
                  f"MultiStep: {train_losses['multi_step']:.4f}")
            print(f"  Val   - Total: {val_losses['total']:.4f}, "
                  f"NextTok: {val_losses['next_token']:.4f}, "
                  f"MultiStep: {val_losses['multi_step']:.4f}")

            # Save history
            self.train_history.append({
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses,
                'lr': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            })

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(save_path / "best_initial.pt")
                print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")

        print("\n" + "="*60)
        print("Initial training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*60)

        return {'history': self.train_history, 'best_val_loss': best_val_loss}

    def walk_forward(
        self,
        test_data: np.ndarray,
        window_size: int = 10080,  # 7 days in minutes
        fine_tune_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        freeze_layers: Optional[int] = None,
        save_dir: str = "checkpoints"
    ) -> List[Dict]:
        """
        Walk-forward testing and fine-tuning.

        Args:
            test_data: OHLC test data (n_samples, 4)
            window_size: Size of each walk-forward window in candles
            fine_tune_epochs: Epochs for fine-tuning each window
            batch_size: Batch size
            learning_rate: Learning rate for fine-tuning
            freeze_layers: Number of initial layers to freeze (None = no freezing)
            save_dir: Directory to save checkpoints

        Returns:
            List of results for each window
        """
        print("\n" + "="*60)
        print("WALK-FORWARD TESTING & FINE-TUNING")
        print("="*60)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Tokenize test data
        print("\nTokenizing test data...")
        token_ids = self.tokenizer.encode(test_data)

        # Calculate number of windows
        n_windows = len(token_ids) // window_size
        print(f"  Test data: {len(token_ids):,} candles")
        print(f"  Window size: {window_size:,} candles")
        print(f"  Number of windows: {n_windows}")

        # Setup optimizer for fine-tuning
        self._setup_optimizer(learning_rate, weight_decay=0.01)

        # Freeze layers if specified
        if freeze_layers is not None and freeze_layers > 0:
            print(f"\nFreezing first {freeze_layers} transformer layers")
            for i, block in enumerate(self.model.transformer_blocks):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False

        # Walk forward through windows
        results = []

        for window_idx in range(n_windows):
            print(f"\n{'='*60}")
            print(f"Window {window_idx+1}/{n_windows}")
            print(f"{'='*60}")

            # Get window data
            start_idx = window_idx * window_size
            end_idx = (window_idx + 1) * window_size
            window_token_ids = token_ids[start_idx:end_idx]

            # Create sequences
            sequence_length = self.model.config.context_length // 4
            X_window, y_window = create_sequences(
                window_token_ids,
                sequence_length,
                stride=sequence_length // 2
            )

            # Create dataloader
            dataset = TensorDataset(
                torch.from_numpy(X_window).long(),
                torch.from_numpy(y_window).long()
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Evaluate before fine-tuning
            print("\nEvaluating on window...")
            losses_before = self.validate(loader)

            # Fine-tune on this window
            print(f"\nFine-tuning for {fine_tune_epochs} epochs...")
            for epoch in range(fine_tune_epochs):
                self.current_epoch = window_idx * fine_tune_epochs + epoch
                train_losses = self.train_epoch(loader, alpha=0.7)
                print(f"  Epoch {epoch+1}: loss={train_losses['total']:.4f}")

            # Evaluate after fine-tuning
            losses_after = self.validate(loader)

            # Save results
            result = {
                'window': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'losses_before': losses_before,
                'losses_after': losses_after,
                'improvement': losses_before['total'] - losses_after['total'],
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            print(f"\nWindow results:")
            print(f"  Loss before: {losses_before['total']:.4f}")
            print(f"  Loss after:  {losses_after['total']:.4f}")
            print(f"  Improvement: {result['improvement']:.4f}")

            # Save checkpoint every N windows
            if (window_idx + 1) % 5 == 0:
                self.save_checkpoint(save_path / f"walk_forward_window_{window_idx+1}.pt")

        # Save final checkpoint
        self.save_checkpoint(save_path / "walk_forward_final.pt")

        # Save results
        with open(save_path / "walk_forward_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*60)
        print("Walk-forward complete!")
        print("="*60)

        return results

    def save_checkpoint(self, filepath: Path):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.config,
            'train_history': self.train_history,
            'walk_forward_history': self.walk_forward_history
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.train_history = checkpoint.get('train_history', [])
        self.walk_forward_history = checkpoint.get('walk_forward_history', [])

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Global step: {self.global_step}")
        print(f"  Current epoch: {self.current_epoch}")


if __name__ == "__main__":
    print("Walk-forward trainer implemented successfully!")
    print("See main training script for usage example.")
