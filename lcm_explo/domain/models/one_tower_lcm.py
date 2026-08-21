from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lcm_explo.domain.models.base_lcm import Normalizer, UnknowNormTypeError, make_norm
from lcm_explo.domain.models.nn import DyT, QKNormedMultiheadAttention


@dataclass
class OneTowerLCMConfig:
    """Configuration for the One-Tower LCM model (DDPM x⁰-prediction, interleaved)."""

    concept_embedding_dim: int = 1024
    hidden_size: int = 2048          # 32 heads × 64 head_dim
    num_attention_heads: int = 32
    num_hidden_layers: int = 12
    intermediate_size: int = 8192    # 4× hidden_size
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    norm_type: Literal["RMSNorm", "DyT"] = "DyT"
    max_seq_len: int = 32
    num_denoising_steps: int = 1000  # T for DDPM schedule
    num_sampling_steps: int = 40     # DDIM trailing steps at inference
    cfg_prob: float = 0.15           # drop_attn probability during training
    cfg_scale: float = 3.0           # guidance scale at inference
    timestep_dim: int = 256          # sinusoidal timestep embedding dim


def _make_norm_ot(config: OneTowerLCMConfig) -> nn.Module:
    """Build norm layer for OneTowerLCMConfig (mirrors make_norm from base_lcm)."""
    match config.norm_type:
        case "RMSNorm":
            return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        case "DyT":
            return DyT(config.hidden_size, init_alpha=1.0)
        case _:
            raise UnknowNormTypeError(config.norm_type)


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class CosineNoiseSchedule(nn.Module):
    """Cosine schedule with zero terminal SNR (Hoogeboom et al. 2023).

    α²_t = cos²((t/T + s)/(1+s)·π/2) / cos²(s/(1+s)·π/2),  s=0.008.
    α²_T forced to 0 (zero terminal SNR).
    """

    alphas_cumprod: Tensor
    sqrt_alphas: Tensor
    sqrt_one_minus_alphas: Tensor

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        t = torch.arange(T + 1).float()
        f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        a2 = (torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2) / f0
        a2 = a2.clamp(0.0, 1.0)
        a2[-1] = 0.0  # zero terminal SNR
        self.register_buffer("alphas_cumprod", a2)
        self.register_buffer("sqrt_alphas", a2.sqrt())
        self.register_buffer("sqrt_one_minus_alphas", (1.0 - a2).clamp(min=0.0).sqrt())

    def q_sample(self, x0: Tensor, t_idx: Tensor) -> tuple[Tensor, Tensor]:
        """DDPM forward: x_t = sqrt(α_t)·x0 + sqrt(1-α_t)·ε. Returns (x_t, ε)."""
        a = self.sqrt_alphas[t_idx].view(-1, 1, 1)
        sg = self.sqrt_one_minus_alphas[t_idx].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        return a * x0 + sg * eps, eps

    def trailing_steps(self, n: int) -> list[int]:
        """DDIM trailing schedule: n evenly-spaced indices over [0, T]."""
        T = len(self.alphas_cumprod) - 1
        step = max(T // n, 1)
        steps = list(range(0, T + 1, step))
        return steps[-n:] if len(steps) > n else steps


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (256-dim) → MLP(Linear→SiLU→Linear) → hidden_size."""

    def __init__(self, dim_out: int, dim_inner: int = 256) -> None:
        super().__init__()
        self.dim_inner = dim_inner
        self.mlp = nn.Sequential(
            nn.Linear(dim_inner, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t: integer timestep indices (B,) → (B, dim_out)."""
        half = self.dim_inner // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)       # (B, dim_inner)
        return self.mlp(emb)                               # (B, dim_out)


# ---------------------------------------------------------------------------
# PreNet (with learned positional embeddings)
# ---------------------------------------------------------------------------

class OneTowerPreNet(nn.Module):
    """concat(normalize(x), t_emb) → Linear(D_SONAR + H, H) + learned PE.

    Learned PE: nn.Embedding(2 * max_seq_len, H).
    Even positions (0, 2, …) → clean context tokens.
    Odd positions  (1, 3, …) → noisy target tokens.
    """

    def __init__(self, config: OneTowerLCMConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(
            config.concept_embedding_dim + config.hidden_size,
            config.hidden_size,
        )
        self.pos_embedding = nn.Embedding(2 * config.max_seq_len, config.hidden_size)

    def forward(self, x_norm: Tensor, t_emb: Tensor, pos_offset: int = 0) -> Tensor:
        """
        Args:
            x_norm:     (B, S, D_SONAR) already normalized
            t_emb:      (B, H) timestep embedding
            pos_offset: starting position index (0 for clean, 1 for noisy)
        Returns:
            (B, S, H)
        """
        S = x_norm.shape[1]
        t_exp = t_emb.unsqueeze(1).expand(-1, S, -1)          # (B, S, H)
        h = self.proj(torch.cat([x_norm, t_exp], dim=-1))     # (B, S, H)
        pos = torch.arange(pos_offset, pos_offset + 2 * S, 2, device=x_norm.device)  # every-other
        return h + self.pos_embedding(pos)                     # (B, S, H)


# ---------------------------------------------------------------------------
# Interleaved attention mask
# ---------------------------------------------------------------------------

def _build_interleaved_mask(S: int, device: torch.device) -> Tensor:
    """Build attention mask for interleaved sequence of length 2S.

    Layout: [c_0, n_1, c_1, n_2, ..., c_{S-1}, n_S]
      - Even position 2k  (clean c_k):   sees c_0, c_2, ..., c_{2k} only.
      - Odd  position 2k+1 (noisy n_k+1): sees c_0, c_2, ..., c_{2k} + self.

    Returns additive mask (1, 1, 2S, 2S): 0.0 = attend, -inf = block.
    """
    total = 2 * S
    mask = torch.full((total, total), float("-inf"), device=device)
    for i in range(total):
        k = i // 2
        for j in range(0, 2 * k + 1, 2):  # even positions up to 2k
            mask[i, j] = 0.0
        if i % 2 == 1:                      # noisy token can also attend to itself
            mask[i, i] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, 2S, 2S)


# ---------------------------------------------------------------------------
# Decoder layer (DyT + QKNormedMultiheadAttention, no cross-attention)
# ---------------------------------------------------------------------------

class OneTowerDecoderLayer(nn.Module):
    """Single-tower decoder layer with QK-normed self-attention and DyT norms."""

    self_attention_layer_norm: nn.LayerNorm | DyT
    feed_forward_layer_norm: nn.LayerNorm | DyT

    def __init__(self, config: OneTowerLCMConfig) -> None:
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
        self.self_attention_layer_norm = _make_norm_ot(config)
        self.feed_forward_layer_norm   = _make_norm_ot(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        # Pre-norm self-attention with interleaved mask
        residual = hidden
        h_norm = self.self_attention_layer_norm(hidden)
        hidden, _ = self.self_attention(h_norm, h_norm, h_norm, attention_mask=mask)
        hidden = residual + self.dropout(hidden)

        # Pre-norm FFN
        residual = hidden
        hidden = self.feed_forward(self.feed_forward_layer_norm(hidden))
        return residual + self.dropout(hidden)


class OneTowerDecoder(nn.Module):
    """Stack of OneTowerDecoderLayers with final norm."""

    def __init__(self, config: OneTowerLCMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [OneTowerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = _make_norm_ot(config)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.layer_norm(hidden)


# ---------------------------------------------------------------------------
# Base pretrained wrapper
# ---------------------------------------------------------------------------

class OneTowerLCMPreTrainedModel(nn.Module):
    config_class = OneTowerLCMConfig
    base_model_prefix = "one_tower_lcm"

    def __init__(self, config: OneTowerLCMConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
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
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def tie_weights(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OneTowerLCM(OneTowerLCMPreTrainedModel):
    """One-Tower LCM (DDPM cosine schedule, x⁰-prediction, interleaved attention, DDIM + CFG)."""

    def __init__(self, config: OneTowerLCMConfig) -> None:
        super().__init__(config)
        self.normalizer         = Normalizer(config.concept_embedding_dim)
        self.noise_schedule     = CosineNoiseSchedule(T=config.num_denoising_steps)
        self.timestep_embedding = TimestepEmbedding(config.hidden_size, config.timestep_dim)
        self.prenet             = OneTowerPreNet(config)
        self.decoder            = OneTowerDecoder(config)
        self.postnet            = nn.Linear(config.hidden_size, config.concept_embedding_dim)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_concepts: Tensor,
        target_concepts: Tensor,
        t_idx: Tensor | None = None,
        drop_attn: bool = False,
    ) -> dict[str, Tensor]:
        """Teacher-forced training with interleaved attention.

        Args:
            input_concepts:  clean SONAR context (B, S, D), raw space
            target_concepts: SONAR targets to denoise (B, S, D), raw space
            t_idx:           integer timestep indices (B,); sampled if None
            drop_attn:       if True, zero out clean context (CFG unconditional)
        Returns:
            dict with keys 'pred_x0', 'x0', 'loss'
        """
        B, S, _ = target_concepts.shape
        x0  = self.normalizer.normalize(target_concepts)   # (B, S, D)
        ctx = self.normalizer.normalize(input_concepts)    # (B, S, D)

        if t_idx is None:
            T = len(self.noise_schedule.alphas_cumprod)
            t_idx = torch.randint(1, T, (B,), device=x0.device)

        x_t, _ = self.noise_schedule.q_sample(x0, t_idx)  # (B, S, D)
        t_emb   = self.timestep_embedding(t_idx)           # (B, H)

        # Build interleaved sequence: [c0, n1, c1, n2, ...]
        ctx_h   = self.prenet(ctx, t_emb, pos_offset=0)    # even positions → (B, S, H)
        noisy_h = self.prenet(x_t, t_emb, pos_offset=1)   # odd  positions → (B, S, H)

        # Interleave to (B, 2S, H)
        interleaved = torch.stack([ctx_h, noisy_h], dim=2).view(B, 2 * S, -1)

        if drop_attn:
            interleaved[:, 0::2, :] = 0.0  # zero clean tokens for CFG

        mask    = _build_interleaved_mask(S, x0.device)   # (1, 1, 2S, 2S)
        out     = self.decoder(interleaved, mask)           # (B, 2S, H)
        pred_x0 = self.postnet(out[:, 1::2, :])            # odd positions → (B, S, D)

        loss = F.mse_loss(pred_x0, x0)
        return {"pred_x0": pred_x0, "x0": x0, "loss": loss}

    # ------------------------------------------------------------------
    # Inference: DDIM + CFG
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, input_concepts: Tensor, steps: int | None = None) -> Tensor:
        """DDIM trailing sampler with CFG. Returns predicted concepts in raw SONAR space.

        Args:
            input_concepts: context SONAR embeddings (B, S, D), raw space
            steps:          DDIM steps; defaults to config.num_sampling_steps
        Returns:
            (B, S, D) in raw SONAR space
        """
        B, S, D = input_concepts.shape
        n_steps  = steps or self.config.num_sampling_steps
        t_steps  = self.noise_schedule.trailing_steps(n_steps)

        x_t = torch.randn(B, S, D, device=input_concepts.device)

        for i in range(len(t_steps) - 1, 0, -1):
            t_c = torch.full((B,), t_steps[i],     device=x_t.device, dtype=torch.long)
            t_p = torch.full((B,), t_steps[i - 1], device=x_t.device, dtype=torch.long)

            # De-normalize x_t back to raw space for forward() interface
            x_raw = self.normalizer.denormalize(x_t)

            cond   = self.forward(input_concepts, x_raw, t_idx=t_c, drop_attn=False)["pred_x0"]
            uncond = self.forward(input_concepts, x_raw, t_idx=t_c, drop_attn=True )["pred_x0"]

            # CFG + guidance rescaling
            pred_x0 = (uncond + self.config.cfg_scale * (cond - uncond)).clamp(-1.0, 1.0)

            # DDIM update (deterministic, eta=0)
            a_c  = self.noise_schedule.sqrt_alphas[t_c].view(-1, 1, 1)
            sg_c = self.noise_schedule.sqrt_one_minus_alphas[t_c].view(-1, 1, 1)
            a_p  = self.noise_schedule.sqrt_alphas[t_p].view(-1, 1, 1)
            sg_p = self.noise_schedule.sqrt_one_minus_alphas[t_p].view(-1, 1, 1)

            eps_pred = (x_t - a_c * pred_x0) / sg_c.clamp(min=1e-8)
            x_t = a_p * pred_x0 + sg_p * eps_pred

        return self.normalizer.denormalize(x_t)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_normalizer_stats(self, path: str | Path) -> None:
        """Load frozen per-dimension normalization statistics from a normalizer.pt artifact."""
        stats = torch.load(path)  # nosec
        self.normalizer.load_stats(stats["mean"], stats["std"])
