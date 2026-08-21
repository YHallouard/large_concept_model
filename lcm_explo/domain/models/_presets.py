from typing import Literal

from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.models.dlcm import DLCMConfig
from lcm_explo.domain.models.one_tower_lcm import OneTowerLCMConfig

LCMSize = Literal["tiny", "small", "base", "large"]
DLCMSize = Literal["tiny", "small"]

# head_dim = 64 throughout (hidden_size / num_attention_heads = 64)
_BASE_LCM_PRESETS: dict[str, dict] = {
    "tiny":  dict(hidden_size=256,  num_hidden_layers=4,  num_attention_heads=4,  intermediate_size=1024),
    "small": dict(hidden_size=512,  num_hidden_layers=8,  num_attention_heads=8,  intermediate_size=2048),
    "base":  dict(hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, intermediate_size=4096),
    "large": dict(hidden_size=2048, num_hidden_layers=24, num_attention_heads=32, intermediate_size=8192),
}
# Approximate parameter counts (concept_embedding_dim=1024):
#   tiny ≈ 4M  ·  small ≈ 26M  ·  base ≈ 150M  ·  large ≈ 1.2B


def base_lcm_config_for_size(
    size: LCMSize,
    max_seq_len: int = 32,
    norm_type: str = "DyT",
) -> BaseLCMConfig:
    return BaseLCMConfig(
        **_BASE_LCM_PRESETS[size],
        max_seq_len=max_seq_len,
        norm_type=norm_type,  # type: ignore[arg-type]
    )


# One-Tower presets — same head_dim=64 rule.
# PreNet projects (D_SONAR + H) → H, so param counts are slightly higher than Base-LCM.
_ONE_TOWER_PRESETS: dict[str, dict] = {
    "tiny":  dict(hidden_size=256,  num_hidden_layers=4,  num_attention_heads=4,  intermediate_size=1024),
    "small": dict(hidden_size=512,  num_hidden_layers=8,  num_attention_heads=8,  intermediate_size=2048),
    "base":  dict(hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, intermediate_size=4096),
    "large": dict(hidden_size=2048, num_hidden_layers=24, num_attention_heads=32, intermediate_size=8192),
}
# Approximate parameter counts (concept_embedding_dim=1024):
#   tiny ≈ 5M  ·  small ≈ 27M  ·  base ≈ 155M  ·  large ≈ 620M
#
# VRAM budget (AdamW fp32, single GPU):
#   tiny  → ~0.08 GB  ·  small → ~0.43 GB  ·  base → ~2.5 GB  ·  large → ~10 GB (needs bf16)


def one_tower_config_for_size(
    size: LCMSize,
    max_seq_len: int = 32,
    norm_type: str = "DyT",
    num_denoising_steps: int = 1000,
    num_sampling_steps: int = 40,
) -> OneTowerLCMConfig:
    return OneTowerLCMConfig(
        **_ONE_TOWER_PRESETS[size],
        max_seq_len=max_seq_len,
        norm_type=norm_type,       # type: ignore[arg-type]
        num_denoising_steps=num_denoising_steps,
        num_sampling_steps=num_sampling_steps,
    )


# DLCM presets — head_dim=64 throughout. The concept backbone (d_concept) holds
# most of the layers/params, mirroring the paper's 10/16/6 encoder/backbone/decoder
# split scaled down for a single 11 GB GPU.
_DLCM_PRESETS: dict[str, dict] = {
    "tiny":  dict(d_token=256, d_concept=512,  d_scan=64,  num_encoder_layers=2, num_backbone_layers=4, num_decoder_layers=2, num_token_heads=4, num_concept_heads=8),
    "small": dict(d_token=512, d_concept=1024, d_scan=128, num_encoder_layers=4, num_backbone_layers=8, num_decoder_layers=2, num_token_heads=8, num_concept_heads=16),
}
# Approximate parameter counts (vocab_size=50257, tied embeddings):
#   tiny ≈ 30M (vocab-dominated; ~17M non-embedding)  ·  small ≈ 150M
#
# VRAM (small, L=1024, mixed precision): ~2.7 GB weights+AdamW, worst-case M=L
# activations; micro-batch 4 × accumulate 8 fits comfortably in 11 GB.


def dlcm_config_for_size(
    size: DLCMSize,
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    target_ratio: float = 4.0,
    boundary_mode: str = "learned",
) -> DLCMConfig:
    return DLCMConfig(
        **_DLCM_PRESETS[size],
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        target_ratio=target_ratio,
        boundary_mode=boundary_mode,  # type: ignore[arg-type]
    )
