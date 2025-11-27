"""
Market GPT with Multi-Asset Support.
Extends base Market GPT with asset and category embeddings.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

# Import base model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from market_gpt import (
    MarketGPTConfig,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerBlock,
    Temporal PositionalEncoding
)


@dataclass
class MultiAssetConfig(MarketGPTConfig):
    """Extended config with multi-asset support."""
    num_assets: int = 20          # Number of different assets
    num_categories: int = 5        # Number of asset categories
    use_category_embedding: bool = True  # Use hierarchical embeddings


class MarketGPTMultiAsset(nn.Module):
    """
    Market GPT with multi-asset support.

    Adds asset-specific and category-specific embeddings
    to condition the model on which asset it's predicting.
    """

    def __init__(self, config: MultiAssetConfig):
        super().__init__()
        self.config = config

        # Token embeddings (same as before)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.d_model // config.n_features)
            for _ in range(config.n_features)
        ])

        # Projection to full d_model
        self.emb_proj = nn.Linear(config.d_model, config.d_model)

        # NUEVO: Asset embeddings
        self.asset_embedding = nn.Embedding(config.num_assets, config.d_model)

        # NUEVO: Category embeddings (hierarchical)
        if config.use_category_embedding:
            self.category_embedding = nn.Embedding(config.num_categories, config.d_model)
        else:
            self.category_embedding = None

        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(config)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output heads
        self.next_token_head = nn.Linear(
            config.d_model,
            config.vocab_size * config.n_features,
            bias=False
        )

        self.multi_step_proj = nn.Linear(config.d_model, config.d_model)
        self.multi_step_head = nn.Linear(
            config.d_model,
            config.n_steps_pred * config.n_features,
            bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MarketGPT Multi-Asset initialized with {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        asset_ids: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        temporal_ids: Optional[torch.Tensor] = None,
        mode: str = "both"
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with asset conditioning.

        Args:
            token_ids: Tokenized OHLC (B, T, 4)
            asset_ids: Asset IDs (B, T) or (B,) if same for all timesteps
            category_ids: Category IDs (B, T) or (B,)
            temporal_ids: Temporal IDs for time-of-day encoding
            mode: Prediction mode

        Returns:
            - next_token_logits: (B, T, vocab_size * 4) or None
            - multi_step_preds: (B, T, n_steps * 4) or None
        """
        B, T, F = token_ids.size()
        assert F == self.config.n_features

        # Embed OHLC tokens
        embeddings = []
        for i in range(self.config.n_features):
            emb = self.token_embeddings[i](token_ids[:, :, i])
            embeddings.append(emb)

        x = torch.cat(embeddings, dim=-1)  # (B, T, d_model)
        x = self.emb_proj(x)

        # Add asset embedding
        if asset_ids.dim() == 1:  # (B,) - same asset for all timesteps
            asset_ids = asset_ids.unsqueeze(1).expand(B, T)

        asset_emb = self.asset_embedding(asset_ids)  # (B, T, d_model)
        x = x + asset_emb

        # Add category embedding if available
        if self.category_embedding is not None and category_ids is not None:
            if category_ids.dim() == 1:
                category_ids = category_ids.unsqueeze(1).expand(B, T)

            category_emb = self.category_embedding(category_ids)
            x = x + category_emb

        # Add positional encoding
        x = self.pos_encoding(x, temporal_ids)

        # Embedding dropout
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output heads
        next_token_logits = None
        multi_step_preds = None

        if mode in ["next_token", "both"]:
            next_token_logits = self.next_token_head(x)

        if mode in ["multi_step", "both"]:
            x_ms = torch.relu(self.multi_step_proj(x))
            multi_step_preds = self.multi_step_head(x_ms)

        return next_token_logits, multi_step_preds

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            "token_embeddings": sum(p.numel() for emb in self.token_embeddings for p in emb.parameters()),
            "asset_embeddings": sum(p.numel() for p in self.asset_embedding.parameters()),
            "category_embeddings": sum(p.numel() for p in self.category_embedding.parameters()) if self.category_embedding else 0,
            "positional": sum(p.numel() for p in self.pos_encoding.parameters()),
            "transformer": sum(p.numel() for block in self.transformer_blocks for p in block.parameters()),
            "output_heads": sum(p.numel() for p in [*self.next_token_head.parameters(), *self.multi_step_head.parameters()]),
            "total": sum(p.numel() for p in self.parameters())
        }
        return counts


if __name__ == "__main__":
    # Test
    config = MultiAssetConfig(
        n_layers=4,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        context_length=128,
        vocab_size=512,
        num_assets=20,
        num_categories=5
    )

    model = MarketGPTMultiAsset(config)

    # Test forward pass
    B, T = 2, 64
    token_ids = torch.randint(0, 512, (B, T, 4))
    asset_ids = torch.randint(0, 20, (B, T))
    category_ids = torch.ones(B, T, dtype=torch.long) * 4  # All crypto

    next_logits, multi_preds = model(
        token_ids,
        asset_ids,
        category_ids,
        mode="both"
    )

    print(f"Input shape: {token_ids.shape}")
    print(f"Asset IDs shape: {asset_ids.shape}")
    print(f"Next-token logits shape: {next_logits.shape}")
    print(f"Multi-step preds shape: {multi_preds.shape}")

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
