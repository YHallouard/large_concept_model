from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcm_explo.domain.models._losses import RMSELoss


@dataclass
class OneTowerLCMConfig:
    """Configuration for the One-Tower LCM model."""

    concept_embedding_dim: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.3
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    num_denoising_steps: int = 50


class OneTowerLCMPreTrainedModel(nn.Module):
    config_class = OneTowerLCMConfig
    base_model_prefix = "one_tower_lcm"

    def __init__(self, config: OneTowerLCMConfig):
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


class OneTowerLCMPreNet(nn.Module):
    def __init__(self, config: OneTowerLCMConfig):
        super().__init__()
        self.input_dense = nn.Linear(config.concept_embedding_dim, config.hidden_size)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concept_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states: torch.Tensor = F.gelu(self.input_dense(concept_embeddings))
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.gelu(self.intermediate_dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class OneTowerLCMPostNet(nn.Module):
    def __init__(self, config: OneTowerLCMConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dense = nn.Linear(config.hidden_size, config.concept_embedding_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:
        intermediate_states = F.gelu(self.intermediate_dense(input_states))
        intermediate_states = self.dropout(intermediate_states)
        intermediate_states = self.layer_norm(intermediate_states)
        concept_embeddings: torch.Tensor = self.output_dense(intermediate_states)
        return concept_embeddings


class OneTowerLCMDecoderLayer(nn.Module):
    def __init__(self, config: OneTowerLCMConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._self_attention_block(hidden_states)
        hidden_states = self._cross_attention_block(hidden_states, encoder_hidden_states)
        hidden_states = self._feed_forward_block(hidden_states)
        return hidden_states

    def _self_attention_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attention_layer_norm(hidden_states)
        hidden_states, _ = self.self_attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _cross_attention_block(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.cross_attention_layer_norm(hidden_states)
        hidden_states, _ = self.cross_attention(hidden_states, encoder_hidden_states, encoder_hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _feed_forward_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OneTowerLCMDecoder(nn.Module):
    def __init__(self, config: OneTowerLCMConfig):
        super().__init__()
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.layers = nn.ModuleList([OneTowerLCMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        noise_embeddings: torch.Tensor = self.noise_embedding(noise_level.unsqueeze(-1))
        hidden_states = hidden_states + noise_embeddings.unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class OneTowerLCM(OneTowerLCMPreTrainedModel):
    def __init__(self, config: OneTowerLCMConfig):
        super().__init__(config)
        self.loss = RMSELoss(100)
        self.prenet = OneTowerLCMPreNet(config)
        self.decoder = OneTowerLCMDecoder(config)
        self.postnet = OneTowerLCMPostNet(config)
        self.register_buffer("timesteps", torch.linspace(0, 1, config.num_denoising_steps))
        self.apply(self._init_weights)

    def _add_noise(self, concept_embeddings: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(concept_embeddings)
        noised_embeddings: torch.Tensor = concept_embeddings + noise_level.view(-1, 1, 1) * noise
        return noised_embeddings

    def _get_noise_level(
        self, batch_size: int, device: torch.device, noise_level: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise_level is None:
            noise_level = torch.rand(batch_size, device=device)
        return noise_level

    def _get_noised_concepts(
        self,
        input_concepts: torch.Tensor,
        noise_level: torch.Tensor,
        target_concepts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_concepts is not None:
            noised_concepts = self._add_noise(target_concepts, noise_level)
        else:
            noised_concepts = input_concepts
        return noised_concepts

    def _get_output_dict(
        self,
        predicted_concepts: torch.Tensor,
        target_concepts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {"predicted_concepts": predicted_concepts}

        if target_concepts is not None:
            outputs["loss"] = self.loss(predicted_concepts, target_concepts)

        return outputs

    def forward(
        self,
        input_concepts: torch.Tensor,
        target_concepts: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        batch_size = input_concepts.shape[0]
        noise_level = self._get_noise_level(batch_size, input_concepts.device, noise_level)

        noised_concepts = self._get_noised_concepts(input_concepts, noise_level, target_concepts)

        encoder_hidden_states = self.prenet(input_concepts)
        decoder_hidden_states = self.prenet(noised_concepts)

        hidden_states = self.decoder(
            decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            noise_level=noise_level,
        )

        predicted_concepts = self.postnet(hidden_states)

        if not return_dict:
            return (predicted_concepts,)

        return self._get_output_dict(predicted_concepts, target_concepts)
