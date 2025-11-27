"""
Diagnostic metrics for crypto training.
Helps identify learning issues.
"""
import torch
import numpy as np
from collections import Counter


def compute_feature_accuracy(logits_last, targets, feature_names=['Open', 'High', 'Low', 'Close']):
    """
    Compute accuracy for each OHLC feature separately.

    Args:
        logits_last: (B, 4, vocab_size) - predictions for last timestep
        targets: (B, 4) - target tokens
        feature_names: Names of features

    Returns:
        dict with accuracy per feature
    """
    accuracies = {}

    for i, name in enumerate(feature_names):
        preds = logits_last[:, i, :].argmax(dim=-1)
        correct = (preds == targets[:, i]).sum().item()
        total = targets[:, i].numel()
        accuracy = 100 * correct / total if total > 0 else 0
        accuracies[name] = accuracy

    return accuracies


def analyze_prediction_distribution(logits_last, targets, vocab_size=512):
    """
    Analyze if model is just predicting the same tokens.

    Args:
        logits_last: (B, 4, vocab_size)
        targets: (B, 4)

    Returns:
        dict with distribution stats
    """
    stats = {}

    for i, name in enumerate(['Open', 'High', 'Low', 'Close']):
        preds = logits_last[:, i, :].argmax(dim=-1).cpu().numpy()

        # Count unique predictions
        unique_preds = np.unique(preds)
        pred_counts = Counter(preds)

        # Most common prediction
        most_common = pred_counts.most_common(1)[0]

        stats[name] = {
            'unique_predictions': len(unique_preds),
            'total_vocab_used_pct': 100 * len(unique_preds) / vocab_size,
            'most_common_token': most_common[0],
            'most_common_freq': most_common[1],
            'most_common_pct': 100 * most_common[1] / len(preds)
        }

    return stats


def compute_per_feature_loss(logits_last, targets, criterion):
    """
    Compute loss for each feature separately.

    Args:
        logits_last: (B, 4, vocab_size)
        targets: (B, 4)
        criterion: loss function

    Returns:
        dict with loss per feature
    """
    losses = {}

    for i, name in enumerate(['Open', 'High', 'Low', 'Close']):
        loss = criterion(logits_last[:, i, :], targets[:, i])
        losses[name] = loss.item()

    return losses


def print_diagnostic_summary(epoch, train_metrics, val_metrics):
    """
    Print comprehensive diagnostic summary.
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC SUMMARY - Epoch {epoch}")
    print(f"{'='*80}")

    # Overall metrics
    print(f"\n📊 Overall Metrics:")
    print(f"  Train Loss: {train_metrics['loss']:.4f}")
    print(f"  Val Loss:   {val_metrics['loss']:.4f}")
    print(f"  Val Accuracy: {val_metrics['accuracy']:.2f}%")

    # Per-feature accuracy
    print(f"\n🎯 Accuracy by Feature:")
    for feature in ['Open', 'High', 'Low', 'Close']:
        acc = val_metrics['feature_accuracy'][feature]
        print(f"  {feature:6s}: {acc:6.2f}%")

    # Per-feature loss
    print(f"\n📉 Loss by Feature:")
    for feature in ['Open', 'High', 'Low', 'Close']:
        loss = val_metrics['feature_loss'][feature]
        print(f"  {feature:6s}: {loss:7.4f}")

    # Prediction distribution
    print(f"\n🔍 Prediction Distribution (Validation):")
    for feature in ['Open', 'High', 'Low', 'Close']:
        dist = val_metrics['pred_distribution'][feature]
        print(f"  {feature}:")
        print(f"    Unique tokens predicted: {dist['unique_predictions']} / {dist['total_vocab_used_pct']:.1f}% of vocab")
        print(f"    Most common: token {dist['most_common_token']} ({dist['most_common_pct']:.1f}% of predictions)")

    print(f"{'='*80}\n")


def is_model_learning(history, window=3):
    """
    Check if model is learning by analyzing recent loss trend.

    Args:
        history: list of losses
        window: number of recent epochs to check

    Returns:
        bool, str (is_learning, message)
    """
    if len(history) < window:
        return True, "Not enough history"

    recent = history[-window:]

    # Check if loss is decreasing
    if recent[-1] < recent[0]:
        decrease = recent[0] - recent[-1]
        pct_decrease = 100 * decrease / recent[0]
        return True, f"Decreasing ({pct_decrease:.2f}% in last {window} epochs)"
    else:
        return False, f"Not decreasing in last {window} epochs"
