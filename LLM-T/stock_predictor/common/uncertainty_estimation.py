"""
Uncertainty estimation for model predictions.
Used to generate confident trading signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Makes multiple forward passes with dropout enabled,
    then computes mean and std of predictions.
    """

    def __init__(self, model: nn.Module, n_samples: int = 20):
        """
        Args:
            model: Neural network model
            n_samples: Number of MC samples to draw
        """
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mode: str = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.

        Args:
            x: Input tensor (B, T, F)
            mode: Prediction mode

        Returns:
            - next_token_mean: Mean prediction for next token
            - next_token_std: Std dev (uncertainty) for next token
            - multi_step_mean: Mean for multi-step prediction
            - multi_step_std: Std dev for multi-step
        """
        # Enable dropout
        self.model.train()

        next_token_preds = []
        multi_step_preds = []

        # Multiple forward passes
        for _ in range(self.n_samples):
            next_logits, multi_preds = self.model(x, mode=mode)

            if next_logits is not None:
                next_token_preds.append(next_logits)
            if multi_preds is not None:
                multi_step_preds.append(multi_preds)

        # Compute mean and std
        if next_token_preds:
            next_token_preds = torch.stack(next_token_preds)
            next_token_mean = next_token_preds.mean(dim=0)
            next_token_std = next_token_preds.std(dim=0)
        else:
            next_token_mean = None
            next_token_std = None

        if multi_step_preds:
            multi_step_preds = torch.stack(multi_step_preds)
            multi_step_mean = multi_step_preds.mean(dim=0)
            multi_step_std = multi_step_preds.std(dim=0)
        else:
            multi_step_mean = None
            multi_step_std = None

        # Back to eval mode
        self.model.eval()

        return next_token_mean, next_token_std, multi_step_mean, multi_step_std


class EnsemblePredictor:
    """
    Ensemble of multiple models for uncertainty estimation.
    """

    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        self.models = models
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mode: str = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.

        Args:
            x: Input tensor
            mode: Prediction mode

        Returns:
            Mean and std for next_token and multi_step predictions
        """
        next_token_preds = []
        multi_step_preds = []

        # Get predictions from all models
        for model in self.models:
            next_logits, multi_preds = model(x, mode=mode)

            if next_logits is not None:
                next_token_preds.append(next_logits)
            if multi_preds is not None:
                multi_step_preds.append(multi_preds)

        # Compute mean and std
        if next_token_preds:
            next_token_preds = torch.stack(next_token_preds)
            next_token_mean = next_token_preds.mean(dim=0)
            next_token_std = next_token_preds.std(dim=0)
        else:
            next_token_mean = None
            next_token_std = None

        if multi_step_preds:
            multi_step_preds = torch.stack(multi_step_preds)
            multi_step_mean = multi_step_preds.mean(dim=0)
            multi_step_std = multi_step_preds.std(dim=0)
        else:
            multi_step_mean = None
            multi_step_std = None

        return next_token_mean, next_token_std, multi_step_mean, multi_step_std


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of probability distribution.
    Higher entropy = higher uncertainty.

    Args:
        probs: Probability tensor (softmax output)

    Returns:
        Entropy values
    """
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


def predictive_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute predictive entropy from logits.

    Args:
        logits: Model logits
        temperature: Temperature for softmax

    Returns:
        Entropy (uncertainty measure)
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return entropy(probs)


class ConfidenceEstimator:
    """
    Estimate confidence from model predictions.
    Maps uncertainty to confidence score in [0, 1].
    """

    def __init__(self, uncertainty_scale: float = 1.0):
        """
        Args:
            uncertainty_scale: Scale factor for uncertainty normalization
        """
        self.uncertainty_scale = uncertainty_scale

    def uncertainty_to_confidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Convert uncertainty (std dev) to confidence score.

        Args:
            uncertainty: Standard deviation of predictions

        Returns:
            Confidence in [0, 1], where 1 = very confident
        """
        # Sigmoid transform: low uncertainty → high confidence
        confidence = 1.0 / (1.0 + uncertainty * self.uncertainty_scale)
        return confidence

    def entropy_to_confidence(self, entropy_value: torch.Tensor, max_entropy: float = None) -> torch.Tensor:
        """
        Convert entropy to confidence score.

        Args:
            entropy_value: Entropy of prediction
            max_entropy: Maximum possible entropy (for normalization)

        Returns:
            Confidence in [0, 1]
        """
        if max_entropy is None:
            # Assume uniform distribution as max entropy
            max_entropy = -np.log(1.0 / entropy_value.shape[-1])

        # Normalize entropy to [0, 1], then invert
        normalized_entropy = entropy_value / max_entropy
        confidence = 1.0 - normalized_entropy
        return confidence.clamp(0, 1)


if __name__ == "__main__":
    # Test
    from market_gpt import MarketGPT, MarketGPTConfig

    # Create dummy model
    config = MarketGPTConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=512,
        context_length=64,
        vocab_size=256,
        dropout=0.1
    )
    model = MarketGPT(config)

    # Dummy input
    x = torch.randint(0, 256, (2, 32, 4))

    # Test MC Dropout
    print("Testing MC Dropout...")
    mc_predictor = MCDropoutPredictor(model, n_samples=10)
    next_mean, next_std, multi_mean, multi_std = mc_predictor.predict_with_uncertainty(x)

    print(f"Next token prediction shape: {next_mean.shape}")
    print(f"Next token uncertainty shape: {next_std.shape}")
    print(f"Uncertainty range: [{next_std.min():.4f}, {next_std.max():.4f}]")

    # Test confidence
    print("\nTesting Confidence Estimator...")
    conf_est = ConfidenceEstimator(uncertainty_scale=1.0)
    confidence = conf_est.uncertainty_to_confidence(next_std)
    print(f"Confidence shape: {confidence.shape}")
    print(f"Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    print("\nUncertainty estimation tests passed!")
