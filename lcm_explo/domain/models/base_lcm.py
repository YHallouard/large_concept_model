from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lcm_explo.domain.models.nn import DyT, QKNormedMultiheadAttention


class UnknowNormTypeError(Exception):
    """Custom exception for unknown normalization types."""

    def __init__(self, norm_type: str) -> None:
        self.message = f"Unknown norm type: {norm_type}"
        super().__init__(self.message)


class StandardScaler(nn.Module):
    """Standard scaler for normalizing tensors."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the scaler.

        Args:
            x: Input tensor of any shape.

        Returns:
            Normalized tensor of the same shape as input.
        """
        if self.training:
            mean = x.mean()
            var = x.var(unbiased=False)
            self.running_mean = 0.1 * mean + 0.9 * self.running_mean
            self.running_var = 0.1 * var + 0.9 * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        return (x - mean) / torch.sqrt(var + self.eps)


@dataclass
class BaseLCMConfig:
    """Configuration for the Base LCM model."""

    concept_embedding_dim: int = 1024
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 12
    intermediate_size: int = 1024 * 4
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    norm_type: Literal["RMSNorm", "DyT"] = "DyT"


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

    def tie_weights(self) -> None:
        pass


class BaseLCMPreNet(nn.Module):
    """Pre-network for the LCM model that normalizes inputs and projects them to hidden dimension."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.scaler = StandardScaler()
        self.proj = nn.Linear(config.concept_embedding_dim, config.hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, StandardScaler]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, concept_embedding_dim)
        Returns:
            Tuple of:
            - Normalized and projected tensor of shape (batch_size, seq_len, hidden_size)
            - Scaler instance for denormalization in PostNet
        """
        x = self.scaler(x)
        x = self.proj(x)
        return x, self.scaler


class BaseLCMPostNet(nn.Module):
    """Post-network for the LCM model that projects hidden states back to input dimension and denormalizes."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size, config.concept_embedding_dim)

    def forward(self, x: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            scaler: StandardScaler instance from PreNet for denormalization
        Returns:
            Denormalized tensor of shape (batch_size, seq_len, concept_embedding_dim)
        """
        x = self.proj(x)
        x = x * torch.sqrt(scaler.running_var + scaler.eps) + scaler.running_mean
        return x


class BaseLCMDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and feed-forward network."""

    self_attention_layer_norm: nn.LayerNorm | DyT
    feed_forward_layer_norm: nn.LayerNorm | DyT

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__()
        self.self_attention = QKNormedMultiheadAttention(
            config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self._initialize_layer_norm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _initialize_layer_norm(self, config: BaseLCMConfig) -> None:
        """Initialize the layer normalization based on the config norm type."""
        if config.norm_type == "RMSNorm":
            self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif config.norm_type == "DyT":
            self.self_attention_layer_norm = DyT(config.hidden_size, init_alpha=0.5)
            self.feed_forward_layer_norm = DyT(config.hidden_size, init_alpha=0.5)
        else:
            raise UnknowNormTypeError(config.norm_type)

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
        # Apply padding mask to attention scores
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        attention_mask = (1.0 - attention_mask) * -10000.0  # Convert mask to large negative values
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
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
    """Base Latent Consistency Model for next-concept prediction."""

    def __init__(self, config: BaseLCMConfig) -> None:
        super().__init__(config)
        self.pre_net = BaseLCMPreNet(config)
        self.decoder = BaseLCMDecoder(config)
        self.post_net = BaseLCMPostNet(config)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of SONAR embeddings, shape (batch_size, seq_len, concept_embedding_dim)
        Returns:
            Predicted next concepts, shape (batch_size, seq_len, concept_embedding_dim)
        """
        x, scaler = self.pre_net(x)
        x = self.decoder(x, padding_mask)
        x = self.post_net(x, scaler)
        return x
