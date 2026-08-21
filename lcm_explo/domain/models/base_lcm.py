from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from lcm_explo.domain.models.nn import DyT, QKNormedMultiheadAttention
from lcm_explo.domain.models.nn.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding


class UnknowNormTypeError(Exception):
    """Custom exception for unknown normalization types."""

    def __init__(self, norm_type: str) -> None:
        self.message = f"Unknown norm type: {norm_type}"
        super().__init__(self.message)


class Normalizer(nn.Module):
    """Per-dimension standardizer with frozen statistics.

    Statistics are fit once on the embedding corpus (see fit_normalizer_task) and loaded
    via load_stats. mean/std are buffers, so they travel with the model state_dict.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def load_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.to(self.mean.device))
        self.std.copy_(std.to(self.std.device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


@dataclass
class BaseLCMConfig:
    """Configuration for the Base LCM model."""

    concept_embedding_dim: int = 1024
    hidden_size: int = 2048
    max_seq_len: int = 32
    num_attention_heads: int = 16
    num_hidden_layers: int = 12
    intermediate_size: int = 1024 * 4
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    norm_type: Literal["RMSNorm", "DyT"] = "DyT"


def make_norm(config: BaseLCMConfig) -> nn.Module:
    """Build the normalization layer matching the config's norm_type."""
    match config.norm_type:
        case "RMSNorm":
            return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        case "DyT":
            return DyT(config.hidden_size, init_alpha=1.0)
        case _:
            raise UnknowNormTypeError(config.norm_type)


class BaseLCMPreTrainedModel(nn.Module):
    config_class = BaseLCMConfig
    base_model_prefix = "base_lcm"

    def __init__(self, config: BaseLCMConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DyT):
            module.alpha.data.fill_(1.0)
            module.gamma.data.fill_(1.0)
            module.beta.data.zero_()

    def tie_weights(self) -> None:
        pass


class BaseLCMPreNet(nn.Module):
    """Projects (already normalized) SONAR concepts to the model hidden dimension."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.concept_embedding_dim, config.hidden_size)
        self.positional_embedding = SinusoidalPositionalEmbedding(config.hidden_size, config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized input tensor of shape (batch_size, seq_len, concept_embedding_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.proj(x)
        x = self.positional_embedding(x)
        return x


class BaseLCMPostNet(nn.Module):
    """Projects hidden states back to the concept dimension (normalized space)."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size, config.concept_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Predicted concepts in normalized space, shape (batch_size, seq_len, concept_embedding_dim)
        """
        return self.proj(x)


class BaseLCMDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and feed-forward network."""

    self_attention_layer_norm: nn.LayerNorm | DyT
    feed_forward_layer_norm: nn.LayerNorm | DyT

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.self_attention = QKNormedMultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.self_attention_layer_norm = make_norm(config)
        self.feed_forward_layer_norm = make_norm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            padding_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            Processed tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.self_attention_layer_norm(hidden_states)
        seq_len = hidden_states.shape[1]
        causal_mask = (
            torch.triu(torch.full((seq_len, seq_len), -10000.0, device=hidden_states.device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, S, S)
        padding_attn_mask = (1.0 - padding_mask.unsqueeze(1).unsqueeze(2)) * -10000.0  # (B, 1, 1, S)
        attention_mask = causal_mask + padding_attn_mask  # (B, 1, S, S)
        hidden_states, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward block
        residual = hidden_states
        hidden_states = self.feed_forward_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BaseLCMDecoder(nn.Module):
    """Decoder-only Transformer for next-concept prediction."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BaseLCMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = make_norm(config)

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Processed tensor of shape (batch_size, seq_len, hidden_size)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BaseLCM(BaseLCMPreTrainedModel):
    """Base Large Concept Model for next-concept prediction."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__(config)
        self.normalizer = Normalizer(config.concept_embedding_dim)
        self.pre_net = BaseLCMPreNet(config)
        self.decoder = BaseLCMDecoder(config)
        self.post_net = BaseLCMPostNet(config)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SONAR embeddings (raw space), shape (batch_size, seq_len, concept_embedding_dim)
        Returns:
            Predicted next concepts in raw SONAR space, shape (batch_size, seq_len, concept_embedding_dim)
        """
        x = self.normalizer.normalize(x)
        x = self.pre_net(x)
        x = self.decoder(x, padding_mask)
        x = self.post_net(x)
        return self.normalizer.denormalize(x)

    def predict(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Alias for forward() — predictions are already in raw SONAR space."""
        return self.forward(x, padding_mask)

    def load_normalizer_stats(self, path: str | Path) -> None:
        """Load frozen per-dimension normalization statistics from a normalizer.pt artifact."""
        stats = torch.load(path)  # nosec
        self.normalizer.load_stats(stats["mean"], stats["std"])
