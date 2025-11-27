"""
Market GPT: Transformer model for financial time series prediction.
Architecture inspired by GPT-2 with adaptations for OHLC candlestick data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketGPTConfig:
    """Configuration for Market GPT model."""

    # Model architecture
    n_layers: int = 12              # Number of transformer blocks
    d_model: int = 768              # Model dimension
    n_heads: int = 12               # Number of attention heads
    d_ff: int = 3072                # Feedforward dimension (4x d_model)

    # Context and vocabulary
    context_length: int = 2048      # Maximum sequence length (2048 candles ≈ 34 hours)
    vocab_size: int = 1024          # Vocabulary size for tokenization

    # OHLC specific
    n_features: int = 4             # Open, High, Low, Close
    n_steps_pred: int = 10          # Number of steps for multi-step prediction

    # Regularization
    dropout: float = 0.1            # Dropout probability
    attention_dropout: float = 0.1   # Attention dropout

    # Training
    use_flash_attention: bool = True  # Use Flash Attention 2 if available
    gradient_checkpointing: bool = False  # Use gradient checkpointing to save memory

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def n_params(self) -> int:
        """Estimate number of parameters."""
        # Embedding
        emb = self.vocab_size * self.d_model + self.context_length * self.d_model

        # Transformer layers
        attn = 4 * self.d_model * self.d_model * self.n_layers  # QKV + projection
        ffn = 2 * self.d_model * self.d_ff * self.n_layers      # Up + down projection
        ln = 2 * self.d_model * self.n_layers                   # LayerNorm (gamma, beta)

        # Output heads
        out = self.d_model * (self.vocab_size * self.n_features + self.n_steps_pred * self.n_features)

        return emb + attn + ffn + ln + out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Implements scaled dot-product attention with multiple heads.
    """

    def __init__(self, config: MarketGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads

        # Q, K, V projections for all heads (in batch)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()  # batch_size, seq_len, d_model

        # Calculate Q, K, V for all heads
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=2)  # Each: (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        out = attn @ v

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Implements: FFN(x) = GELU(xW1 + b1)W2 + b2
    """

    def __init__(self, config: MarketGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """

    def __init__(self, config: MarketGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.ln1(x))

        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))

        return x


class TemporalPositionalEncoding(nn.Module):
    """
    Positional encoding with temporal awareness.
    Combines sinusoidal encoding with learnable time-based features.
    """

    def __init__(self, config: MarketGPTConfig):
        super().__init__()
        self.d_model = config.d_model

        # Sinusoidal positional encoding
        pe = torch.zeros(config.context_length, config.d_model)
        position = torch.arange(0, config.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Learnable temporal embeddings (for capturing patterns like hour-of-day, day-of-week)
        self.temporal_emb = nn.Embedding(24 * 7, config.d_model)  # Hour of week

    def forward(self, x: torch.Tensor, temporal_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            temporal_ids: Optional temporal IDs (B, T) for hour-of-week

        Returns:
            x + positional encoding
        """
        B, T, _ = x.size()

        # Add sinusoidal encoding
        x = x + self.pe[:T, :].unsqueeze(0)

        # Add temporal encoding if provided
        if temporal_ids is not None:
            x = x + self.temporal_emb(temporal_ids)

        return x


class MarketGPT(nn.Module):
    """
    Market GPT: GPT-style Transformer for financial time series prediction.

    Predicts future OHLC values in two modes:
    1. Next-token mode: Autoregressive next candle prediction
    2. Multi-step mode: Direct prediction of next N candles
    """

    def __init__(self, config: MarketGPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings (separate for each OHLC feature)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.d_model // config.n_features)
            for _ in range(config.n_features)
        ])

        # Projection to full d_model
        self.emb_proj = nn.Linear(config.d_model, config.d_model)

        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(config)

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output heads
        # 1. Next-token head (classification)
        self.next_token_head = nn.Linear(
            config.d_model,
            config.vocab_size * config.n_features,
            bias=False
        )

        # 2. Multi-step head (regression)
        self.multi_step_proj = nn.Linear(config.d_model, config.d_model)
        self.multi_step_head = nn.Linear(
            config.d_model,
            config.n_steps_pred * config.n_features,
            bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MarketGPT initialized with {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights with small random values."""
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
        temporal_ids: Optional[torch.Tensor] = None,
        mode: str = "both"
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            token_ids: Tokenized OHLC values (B, T, 4) - integers in [0, vocab_size)
            temporal_ids: Optional temporal IDs (B, T) for hour-of-week encoding
            mode: Prediction mode - "next_token", "multi_step", or "both"

        Returns:
            - next_token_logits: (B, T, vocab_size * 4) if mode in ["next_token", "both"]
            - multi_step_preds: (B, T, n_steps * 4) if mode in ["multi_step", "both"]
        """
        B, T, F = token_ids.size()  # batch, seq_len, n_features (4)
        assert F == self.config.n_features, f"Expected {self.config.n_features} features, got {F}"

        # Embed each OHLC feature separately
        embeddings = []
        for i in range(self.config.n_features):
            emb = self.token_embeddings[i](token_ids[:, :, i])  # (B, T, d_model//4)
            embeddings.append(emb)

        # Concatenate embeddings
        x = torch.cat(embeddings, dim=-1)  # (B, T, d_model)

        # Project to d_model if needed
        x = self.emb_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x, temporal_ids)

        # Embedding dropout
        x = self.emb_dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Generate outputs based on mode
        next_token_logits = None
        multi_step_preds = None

        if mode in ["next_token", "both"]:
            next_token_logits = self.next_token_head(x)  # (B, T, vocab_size * 4)

        if mode in ["multi_step", "both"]:
            x_ms = F.relu(self.multi_step_proj(x))
            multi_step_preds = self.multi_step_head(x_ms)  # (B, T, n_steps * 4)

        return next_token_logits, multi_step_preds

    def generate(
        self,
        token_ids: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation of future candles.

        Args:
            token_ids: Initial context (B, T, 4)
            n_steps: Number of steps to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use all logits)

        Returns:
            Generated token IDs (B, T + n_steps, 4)
        """
        self.eval()

        for _ in range(n_steps):
            # Get predictions for next token
            # Take last context_length tokens if sequence is too long
            idx_cond = token_ids[:, -self.config.context_length:, :]

            # Forward pass
            next_token_logits, _ = self.forward(idx_cond, mode="next_token")

            # Take logits for last position
            logits = next_token_logits[:, -1, :]  # (B, vocab_size * 4)

            # Reshape to (B, 4, vocab_size)
            logits = logits.view(-1, self.config.n_features, self.config.vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.view(-1, self.config.vocab_size), num_samples=1)
            next_token = next_token.view(-1, self.config.n_features)  # (B, 4)

            # Append to sequence
            token_ids = torch.cat([token_ids, next_token.unsqueeze(1)], dim=1)

        return token_ids

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            "embeddings": sum(p.numel() for emb in self.token_embeddings for p in emb.parameters()),
            "positional": sum(p.numel() for p in self.pos_encoding.parameters()),
            "transformer": sum(p.numel() for block in self.transformer_blocks for p in block.parameters()),
            "output_heads": sum(p.numel() for p in [*self.next_token_head.parameters(),
                                                     *self.multi_step_head.parameters()]),
            "total": sum(p.numel() for p in self.parameters())
        }
        return counts


def create_market_gpt(
    size: str = "small",
    **kwargs
) -> MarketGPT:
    """
    Create a MarketGPT model with predefined size configurations.

    Args:
        size: Model size - "small" (~100M), "medium" (~300M), "large" (~500M)
        **kwargs: Override config parameters

    Returns:
        MarketGPT model
    """
    configs = {
        "small": MarketGPTConfig(
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_ff=3072,
        ),
        "medium": MarketGPTConfig(
            n_layers=24,
            d_model=1024,
            n_heads=16,
            d_ff=4096,
        ),
        "large": MarketGPTConfig(
            n_layers=36,
            d_model=1280,
            n_heads=20,
            d_ff=5120,
        )
    }

    if size not in configs:
        raise ValueError(f"Size must be one of {list(configs.keys())}, got {size}")

    config = configs[size]

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MarketGPT(config)


if __name__ == "__main__":
    # Test model creation
    print("="*60)
    print("Testing MarketGPT Model")
    print("="*60)

    # Create model
    config = MarketGPTConfig()
    model = MarketGPT(config)

    print(f"\nConfiguration:")
    print(f"  Layers: {config.n_layers}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Context length: {config.context_length}")
    print(f"  Vocab size: {config.vocab_size}")

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128

    # Dummy input (tokenized OHLC)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len, 4))

    print(f"\nTesting forward pass:")
    print(f"  Input shape: {token_ids.shape}")

    # Forward pass
    next_token_logits, multi_step_preds = model(token_ids, mode="both")

    print(f"  Next-token logits shape: {next_token_logits.shape}")
    print(f"  Multi-step predictions shape: {multi_step_preds.shape}")

    # Test generation
    print(f"\nTesting autoregressive generation:")
    initial_context = token_ids[:, :64, :]
    generated = model.generate(initial_context, n_steps=10, temperature=0.8)
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated {generated.shape[1] - initial_context.shape[1]} new candles")

    print("\n" + "="*60)
    print("Model test completed successfully!")
    print("="*60)
