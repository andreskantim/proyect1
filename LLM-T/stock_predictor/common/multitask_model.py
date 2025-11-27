"""
Multi-task transformer model for market prediction.

Architecture:
1. Input: OHLC sequences (batch, seq_len, 4)
2. Transformer encoder
3. Two prediction heads:
   - Direction head: 3-class classification (DOWN/FLAT/UP)
   - Magnitude head: 4 binary classifications (one per horizon)
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiTaskTransformer(nn.Module):
    """
    Multi-task transformer for market prediction.

    Predicts:
    1. Direction of next candle (3 classes)
    2. Magnitude across 4 horizons (4 binary tasks)
    """

    def __init__(
        self,
        n_features: int = 4,  # OHLC
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        context_length: int = 128,
        n_direction_classes: int = 3,
        n_magnitude_tasks: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.context_length = context_length
        self.n_direction_classes = n_direction_classes
        self.n_magnitude_tasks = n_magnitude_tasks

        # Input projection: OHLC (4 features) -> d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, context_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Prediction heads
        # Direction head: predict 3 classes (DOWN/FLAT/UP)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_direction_classes)
        )

        # Magnitude head: predict 4 binary values (one per horizon)
        # Each binary: does price exceed 2σ Bollinger in this horizon?
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_magnitude_tasks)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features) - OHLC sequences

        Returns:
            Dictionary with:
                - 'direction': (batch, n_direction_classes) - logits for direction
                - 'magnitude': (batch, n_magnitude_tasks) - logits for each horizon
        """
        # Input projection: (B, T, 4) -> (B, T, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Use last timestep for prediction
        x_last = x[:, -1, :]  # (B, d_model)

        # Prediction heads
        direction_logits = self.direction_head(x_last)  # (B, 3)
        magnitude_logits = self.magnitude_head(x_last)  # (B, 4)

        return {
            'direction': direction_logits,
            'magnitude': magnitude_logits
        }

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Loss = w_direction * direction_loss + w_magnitude * magnitude_loss
    """

    def __init__(
        self,
        w_direction: float = 1.0,
        w_magnitude: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.w_direction = w_direction
        self.w_magnitude = w_magnitude

        # Direction: 3-class classification with optional label smoothing
        self.direction_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Magnitude: binary classification per horizon
        self.magnitude_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            predictions: Dictionary with 'direction' and 'magnitude' logits
            targets: Dictionary with 'direction' and 'magnitude' targets

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Direction loss: cross-entropy
        direction_loss = self.direction_criterion(
            predictions['direction'],  # (B, 3)
            targets['direction']       # (B,)
        )

        # Magnitude loss: binary cross-entropy per horizon
        magnitude_loss = self.magnitude_criterion(
            predictions['magnitude'],  # (B, 4)
            targets['magnitude'].float()  # (B, 4) - convert to float
        )

        # Combined loss
        total_loss = (
            self.w_direction * direction_loss +
            self.w_magnitude * magnitude_loss
        )

        loss_dict = {
            'direction': direction_loss.item(),
            'magnitude': magnitude_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Dictionary with 'direction' and 'magnitude' logits
        targets: Dictionary with 'direction' and 'magnitude' targets

    Returns:
        Dictionary with metrics
    """
    # Direction accuracy
    direction_pred = predictions['direction'].argmax(dim=1)  # (B,)
    direction_target = targets['direction']  # (B,)
    direction_acc = (direction_pred == direction_target).float().mean().item()

    # Magnitude accuracy (per horizon and average)
    magnitude_pred = (torch.sigmoid(predictions['magnitude']) > 0.5).long()  # (B, 4)
    magnitude_target = targets['magnitude']  # (B, 4)
    magnitude_acc = (magnitude_pred == magnitude_target).float().mean(dim=0)  # (4,)

    metrics = {
        'direction_acc': direction_acc * 100,
        'magnitude_1w_acc': magnitude_acc[0].item() * 100,
        'magnitude_2w_acc': magnitude_acc[1].item() * 100,
        'magnitude_1m_acc': magnitude_acc[2].item() * 100,
        'magnitude_2m_acc': magnitude_acc[3].item() * 100,
        'magnitude_avg_acc': magnitude_acc.mean().item() * 100
    }

    return metrics


if __name__ == "__main__":
    # Test model
    print("="*70)
    print("Testing Multi-Task Transformer")
    print("="*70)

    # Model configuration
    config = {
        'n_features': 4,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 512,
        'context_length': 128,
        'n_direction_classes': 3,
        'n_magnitude_tasks': 4,
        'dropout': 0.1
    }

    model = MultiTaskTransformer(**config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 16
    seq_len = 128
    x = torch.randn(batch_size, seq_len, 4)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        outputs = model(x)

    print(f"\nOutput shapes:")
    print(f"  direction: {outputs['direction'].shape}")
    print(f"  magnitude: {outputs['magnitude'].shape}")

    # Test loss
    print(f"\n{'='*70}")
    print("Testing Loss Function")
    print('='*70)

    targets = {
        'direction': torch.randint(0, 3, (batch_size,)),
        'magnitude': torch.randint(0, 2, (batch_size, 4))
    }

    criterion = MultiTaskLoss(w_direction=1.0, w_magnitude=1.0)
    loss, loss_dict = criterion(outputs, targets)

    print(f"\nLosses:")
    print(f"  Direction: {loss_dict['direction']:.4f}")
    print(f"  Magnitude: {loss_dict['magnitude']:.4f}")
    print(f"  Total:     {loss_dict['total']:.4f}")

    # Test metrics
    print(f"\n{'='*70}")
    print("Testing Metrics")
    print('='*70)

    metrics = compute_metrics(outputs, targets)
    print(f"\nMetrics:")
    print(f"  Direction accuracy: {metrics['direction_acc']:.2f}%")
    print(f"  Magnitude accuracies:")
    print(f"    1w: {metrics['magnitude_1w_acc']:.2f}%")
    print(f"    2w: {metrics['magnitude_2w_acc']:.2f}%")
    print(f"    1m: {metrics['magnitude_1m_acc']:.2f}%")
    print(f"    2m: {metrics['magnitude_2m_acc']:.2f}%")
    print(f"    Avg: {metrics['magnitude_avg_acc']:.2f}%")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print('='*70)
