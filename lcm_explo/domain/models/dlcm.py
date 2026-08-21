"""Dynamic Large Concept Model (DLCM).

Hierarchical token-level next-token predictor (ByteDance, arXiv:2512.24617):
encoder -> learned dynamic segmentation + mean-pooling -> concept-level backbone
-> decoder with causal concept cross-attention -> LM head. See docs/DLCM.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcm_explo.domain.models.nn.boundary_detector import BoundaryDetector, BoundaryMode
from lcm_explo.domain.models.nn.causal_concept_cross_attention import CausalConceptCrossAttention
from lcm_explo.domain.models.nn.concept_smoothing import CausalConceptSmoothing
from lcm_explo.domain.models.nn.qk_rmsnorm_attention import QKRMSNormSDPAAttention
from lcm_explo.domain.models.nn.segment_pooling import SegmentMeanPooling, segment_ids_from_boundaries
from lcm_explo.domain.models.nn.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding


@dataclass
class DLCMConfig:
    """Configuration for the Dynamic Large Concept Model."""

    vocab_size: int = 50257
    d_token: int = 512
    d_concept: int = 1024
    d_scan: int = 128
    num_encoder_layers: int = 4
    num_backbone_layers: int = 8
    num_decoder_layers: int = 2
    num_token_heads: int = 8
    num_concept_heads: int = 16
    ffn_multiplier: int = 4
    max_seq_len: int = 1024
    target_ratio: float = 4.0
    boundary_temperature: float = 0.5
    boundary_mode: BoundaryMode = "learned"
    boundary_threshold: float = 0.5
    use_smoothing: bool = True
    smoothing_kernel_size: int = 3
    dropout: float = 0.0
    tie_embeddings: bool = True
    initializer_range: float = 0.02


@dataclass
class DLCMOutput:
    """Forward outputs of the DLCM."""

    logits: torch.Tensor  # (B, L, vocab_size)
    boundary_probs: torch.Tensor  # (B, L), continuous p_t
    boundaries: torch.Tensor  # (B, L), hard b_t in {0, 1}
    num_segments: torch.Tensor  # (B,) number of concepts per sequence


class DLCMTransformerLayer(nn.Module):
    """Pre-RMSNorm transformer block with QK-RMSNorm SDPA self-attention."""

    def __init__(self, d_model: int, num_heads: int, ffn_multiplier: int, dropout: float = 0.0) -> None:
        super().__init__()
        head_dim = d_model // num_heads
        self.attention = QKRMSNormSDPAAttention(
            q_dim=d_model, kv_dim=d_model, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )
        self.attention_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False
    ) -> torch.Tensor:
        residual = x
        normed = self.attention_norm(x)
        x = residual + self.dropout(self.attention(normed, normed, normed, attn_mask=attn_mask, is_causal=is_causal))

        residual = x
        x = residual + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DLCMEncoder(nn.Module):
    """Token embedding + causal transformer producing fine-grained representations."""

    def __init__(self, config: DLCMConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_token)
        self.positional_embedding = SinusoidalPositionalEmbedding(config.d_token, config.max_seq_len)
        self.layers = nn.ModuleList(
            DLCMTransformerLayer(config.d_token, config.num_token_heads, config.ffn_multiplier, config.dropout)
            for _ in range(config.num_encoder_layers)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.positional_embedding(self.embedding(input_ids))
        for layer in self.layers:
            h = layer(h, is_causal=True)
        return h


class DLCMSegmenter(nn.Module):
    """Learned boundary detection, mean-pooling and up-projection to concepts."""

    def __init__(self, config: DLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.boundary_detector = BoundaryDetector(config.d_token, config.d_scan, mode=config.boundary_mode)
        self.pooling = SegmentMeanPooling()
        self.w_up = nn.Linear(config.d_token, config.d_concept)

    def forward(
        self, h: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.boundary_detector(h)
        if training:
            b = self.boundary_detector.sample(p, temperature=self.config.boundary_temperature)
        else:
            b = self.boundary_detector.threshold(p, tau=self.config.boundary_threshold)

        seg_id, num_segments = segment_ids_from_boundaries(b)
        pooled, concept_mask = self.pooling(h, seg_id, num_segments)
        concepts = self.w_up(pooled)
        return concepts, concept_mask, seg_id, num_segments, p, b


class DLCMBackbone(nn.Module):
    """High-capacity causal transformer performing reasoning over concepts."""

    def __init__(self, config: DLCMConfig) -> None:
        super().__init__()
        self.positional_embedding = SinusoidalPositionalEmbedding(config.d_concept, config.max_seq_len)
        self.layers = nn.ModuleList(
            DLCMTransformerLayer(config.d_concept, config.num_concept_heads, config.ffn_multiplier, config.dropout)
            for _ in range(config.num_backbone_layers)
        )
        self.norm = nn.RMSNorm(config.d_concept)

    def forward(self, concepts: torch.Tensor, concept_mask: torch.Tensor) -> torch.Tensor:
        num_concepts = concepts.size(1)
        causal = torch.tril(torch.ones(num_concepts, num_concepts, dtype=torch.bool, device=concepts.device))
        key_padding = concept_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
        # (B, 1, M, M): attend to non-future, non-padding concepts. Concept 0 is
        # always valid and causally reachable, so no row is fully masked.
        attn_mask = causal.unsqueeze(0).unsqueeze(0) & key_padding

        z = self.positional_embedding(concepts)
        for layer in self.layers:
            z = layer(z, attn_mask=attn_mask)
        return self.norm(z)


class DLCMDecoder(nn.Module):
    """Reconstructs token predictions via causal concept cross-attention."""

    def __init__(self, config: DLCMConfig) -> None:
        super().__init__()
        self.use_smoothing = config.use_smoothing
        if config.use_smoothing:
            self.smoothing = CausalConceptSmoothing(config.d_concept, config.smoothing_kernel_size)
        self.cross_attention = CausalConceptCrossAttention(
            d_token=config.d_token,
            d_concept=config.d_concept,
            num_heads=config.num_token_heads,
            head_dim=config.d_token // config.num_token_heads,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList(
            DLCMTransformerLayer(config.d_token, config.num_token_heads, config.ffn_multiplier, config.dropout)
            for _ in range(config.num_decoder_layers)
        )
        self.norm = nn.RMSNorm(config.d_token)

    def forward(self, h: torch.Tensor, concepts: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        z = self.smoothing(concepts) if self.use_smoothing else concepts
        h = self.cross_attention(h, z, seg_id)
        for layer in self.layers:
            h = layer(h, is_causal=True)
        return self.norm(h)


class DLCM(nn.Module):
    """Dynamic Large Concept Model for hierarchical next-token prediction."""

    config_class = DLCMConfig
    base_model_prefix = "dlcm"

    def __init__(self, config: DLCMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DLCMEncoder(config)
        self.segmenter = DLCMSegmenter(config)
        self.backbone = DLCMBackbone(config)
        self.decoder = DLCMDecoder(config)
        self.lm_head = nn.Linear(config.d_token, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.encoder.embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> DLCMOutput:
        h = self.encoder(input_ids)
        concepts, concept_mask, seg_id, num_segments, p, b = self.segmenter(h, self.training)
        z = self.backbone(concepts, concept_mask)
        h = self.decoder(h, z, seg_id)
        logits = self.lm_head(h)
        return DLCMOutput(logits=logits, boundary_probs=p, boundaries=b, num_segments=num_segments)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        top_k: int | None = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy / top-k sampling with a naive full re-encode per step.

        Correct because every stage is causal: boundaries already placed do not
        move as the sequence grows. A concept/KV cache (the open segment must be
        re-pooled each step) is deferred.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.max_seq_len :]
            logits = self.forward(window).logits[:, -1, :]  # (B, vocab)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        if was_training:
            self.train()
        return input_ids

    def segment(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hard boundary indicators (B, L) for qualitative inspection."""
        self.eval()
        with torch.no_grad():
            h = self.encoder(input_ids)
            p = self.segmenter.boundary_detector(h)
            return self.segmenter.boundary_detector.threshold(p, tau=self.config.boundary_threshold)
