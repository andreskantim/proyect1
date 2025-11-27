"""
Training progress monitoring with ETA estimation.
"""

import time
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, Optional
import json
from pathlib import Path


class TrainingMonitor:
    """
    Monitor training progress with real-time ETA and metrics.
    """

    def __init__(
        self,
        total_steps: int,
        log_file: str = 'training_progress.log',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.log_file = log_file
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Métricas
        self.loss_history = []
        self.step_times = []
        self.best_loss = float('inf')

        # Progress bar
        self.pbar = tqdm(total=total_steps, desc="Training", ncols=100)

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("timestamp,step,loss,eta_seconds,progress_pct\n")

    def update(self, loss: float, step: int = 1):
        """Update progress with new loss value."""
        self.current_step += step
        self.loss_history.append(loss)

        # Timing
        current_time = time.time()
        elapsed = current_time - self.start_time

        if self.current_step > 0:
            avg_step_time = elapsed / self.current_step
        else:
            avg_step_time = 0

        # ETA
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = avg_step_time * remaining_steps
        eta = timedelta(seconds=int(eta_seconds))

        # Progress
        progress_pct = self.current_step / self.total_steps * 100

        # Update progress bar
        self.pbar.update(step)
        self.pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'best': f'{self.best_loss:.4f}',
            'ETA': str(eta),
            '%': f'{progress_pct:.1f}'
        })

        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{self.current_step},{loss},{eta_seconds},{progress_pct}\n")

    def get_stats(self) -> Dict:
        """Get current training statistics."""
        elapsed = time.time() - self.start_time
        progress_pct = self.current_step / self.total_steps * 100

        if self.current_step > 0:
            avg_step_time = elapsed / self.current_step
        else:
            avg_step_time = 0

        remaining_steps = self.total_steps - self.current_step
        eta_seconds = avg_step_time * remaining_steps

        recent_losses = self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_pct': progress_pct,
            'elapsed_time': str(timedelta(seconds=int(elapsed))),
            'elapsed_seconds': elapsed,
            'eta': str(timedelta(seconds=int(eta_seconds))),
            'eta_seconds': eta_seconds,
            'avg_loss_recent': avg_loss,
            'current_loss': self.loss_history[-1] if self.loss_history else 0,
            'best_loss': self.best_loss,
            'steps_per_second': self.current_step / elapsed if elapsed > 0 else 0
        }

    def save_stats(self, filepath: Optional[str] = None):
        """Save current statistics to JSON."""
        if filepath is None:
            filepath = self.checkpoint_dir / 'training_stats.json'

        stats = self.get_stats()
        stats['timestamp'] = datetime.now().isoformat()
        stats['loss_history'] = self.loss_history

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

    def print_summary(self):
        """Print training summary."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("TRAINING PROGRESS SUMMARY")
        print("="*70)
        print(f"Progress:        {stats['current_step']:,} / {stats['total_steps']:,} steps ({stats['progress_pct']:.1f}%)")
        print(f"Elapsed time:    {stats['elapsed_time']}")
        print(f"ETA:             {stats['eta']}")
        print(f"Speed:           {stats['steps_per_second']:.2f} steps/sec")
        print(f"Current loss:    {stats['current_loss']:.6f}")
        print(f"Best loss:       {stats['best_loss']:.6f}")
        print(f"Avg loss (100):  {stats['avg_loss_recent']:.6f}")
        print("="*70 + "\n")

    def close(self):
        """Close progress bar and save final stats."""
        self.pbar.close()
        self.save_stats()
        self.print_summary()


class EpochMonitor:
    """
    Simpler monitor for epoch-based training.
    """

    def __init__(self, total_epochs: int, steps_per_epoch: int):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.start_time = time.time()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def start_epoch(self, epoch: int):
        """Start a new epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        print(f"{'='*70}")

    def end_epoch(self, train_loss: float, val_loss: float):
        """End current epoch with losses."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        epoch_time = time.time() - self.epoch_start_time
        elapsed = time.time() - self.start_time

        # ETA
        avg_epoch_time = elapsed / (self.current_epoch + 1)
        remaining_epochs = self.total_epochs - (self.current_epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = timedelta(seconds=int(eta_seconds))

        # Check if best
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        print(f"\nEpoch {self.current_epoch + 1} complete:")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss:   {val_loss:.6f} {'✓ BEST' if is_best else ''}")
        print(f"  Epoch time: {timedelta(seconds=int(epoch_time))}")
        print(f"  ETA: {eta}")

    def get_summary(self) -> Dict:
        """Get training summary."""
        elapsed = time.time() - self.start_time

        return {
            'total_epochs': self.total_epochs,
            'completed_epochs': self.current_epoch + 1,
            'elapsed_time': str(timedelta(seconds=int(elapsed))),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None
        }


if __name__ == "__main__":
    # Test
    import random

    monitor = TrainingMonitor(total_steps=1000)

    for i in range(1000):
        loss = 1.0 / (i + 1) + random.random() * 0.1
        monitor.update(loss)

        if i % 100 == 0:
            monitor.print_summary()

        time.sleep(0.01)

    monitor.close()
