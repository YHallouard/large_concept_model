from __future__ import annotations

import uuid
from functools import singledispatch
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from lcm_explo.domain.models._presets import (
    DLCMSize,
    LCMSize,
    base_lcm_config_for_size,
    dlcm_config_for_size,
    one_tower_config_for_size,
)
from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.models.dlcm import DLCMConfig
from lcm_explo.domain.models.one_tower_lcm import OneTowerLCMConfig
from lcm_explo.utils._settings import settings
from lcm_explo.utils.checkpoints import CheckpointStorageConfig, S3CheckpointStorage

# ---------------------------------------------------------------------------
# Init strategies
# ---------------------------------------------------------------------------


class NewRunInit(BaseModel):
    init_type: Literal["new"] = "new"


class ResumeRunInit(BaseModel):
    init_type: Literal["resume"] = "resume"
    run_id: str
    slot: Literal["best", "last"] = "last"


TrainingInitConfig = Annotated[NewRunInit | ResumeRunInit, Field(discriminator="init_type")]


@singledispatch
def resolve_run_id(init: object) -> str:
    raise TypeError(f"Unknown init type: {type(init)}")


@resolve_run_id.register(NewRunInit)
def _(init: NewRunInit) -> str:
    return str(uuid.uuid4())[:8]


@resolve_run_id.register(ResumeRunInit)
def _(init: ResumeRunInit) -> str:
    return init.run_id


# ---------------------------------------------------------------------------
# Model spec — discriminated union for preset sizes vs. full custom config
# ---------------------------------------------------------------------------


class PresetModelSpec(BaseModel):
    spec_type: Literal["preset"] = "preset"
    size: LCMSize = "base"
    max_seq_len: int = 32
    norm_type: Literal["RMSNorm", "DyT"] = "DyT"


class CustomModelSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    spec_type: Literal["custom"] = "custom"
    config: BaseLCMConfig


ModelSpec = Annotated[PresetModelSpec | CustomModelSpec, Field(discriminator="spec_type")]


@singledispatch
def resolve_base_lcm_config(spec: object) -> BaseLCMConfig:
    raise TypeError(f"Unknown model spec type: {type(spec)}")


@resolve_base_lcm_config.register(PresetModelSpec)
def _(spec: PresetModelSpec) -> BaseLCMConfig:
    return base_lcm_config_for_size(spec.size, spec.max_seq_len, spec.norm_type)


@resolve_base_lcm_config.register(CustomModelSpec)
def _(spec: CustomModelSpec) -> BaseLCMConfig:
    return spec.config


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    embeddings_dir: Path
    sequence_length: int = 32
    stride: int = 16


class TrainingConfig(BaseModel):
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000
    min_lr: float = 1e-6
    batch_size: int = 8
    checkpoint_every_n_steps: int = 500
    seed: int = 42
    vram_budget_gb: float | None = None
    # Ex : 11.0 pour RTX 2080 Ti, 24.0 pour RTX 4090, None = pas de limite.
    # Quand n_params × 16 octets dépasse ce budget, DeepSpeed ZeRO-2 optimizer
    # offload est activé automatiquement (GPU uniquement, bf16-mixed requis).


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class TrainBaseLCMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init: TrainingInitConfig = Field(default_factory=NewRunInit)
    data: DataConfig
    model: ModelSpec = Field(default_factory=PresetModelSpec)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint_storage: CheckpointStorageConfig = Field(default_factory=S3CheckpointStorage)
    mlflow_tracking_uri: str = settings.MLFLOW_TRACKING_URI
    experiment_name: str = "base-lcm"


# ---------------------------------------------------------------------------
# One-Tower model spec — mirrors Base-LCM pattern
# ---------------------------------------------------------------------------


class PresetOneTowerModelSpec(BaseModel):
    spec_type: Literal["preset"] = "preset"
    size: LCMSize = "base"
    max_seq_len: int = 32
    norm_type: Literal["RMSNorm", "DyT"] = "DyT"
    num_denoising_steps: int = 1000
    num_sampling_steps: int = 40


class CustomOneTowerModelSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    spec_type: Literal["custom"] = "custom"
    config: OneTowerLCMConfig


OneTowerModelSpec = Annotated[
    PresetOneTowerModelSpec | CustomOneTowerModelSpec,
    Field(discriminator="spec_type"),
]


@singledispatch
def resolve_one_tower_config(spec: object) -> OneTowerLCMConfig:
    raise TypeError(f"Unknown One-Tower model spec type: {type(spec)}")


@resolve_one_tower_config.register(PresetOneTowerModelSpec)
def _(spec: PresetOneTowerModelSpec) -> OneTowerLCMConfig:
    return one_tower_config_for_size(
        spec.size,
        spec.max_seq_len,
        spec.norm_type,
        spec.num_denoising_steps,
        spec.num_sampling_steps,
    )


@resolve_one_tower_config.register(CustomOneTowerModelSpec)
def _(spec: CustomOneTowerModelSpec) -> OneTowerLCMConfig:
    return spec.config


# ---------------------------------------------------------------------------
# One-Tower training config
# ---------------------------------------------------------------------------


class TrainOneTowerLCMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init: TrainingInitConfig = Field(default_factory=NewRunInit)
    data: DataConfig
    model: OneTowerModelSpec = Field(default_factory=PresetOneTowerModelSpec)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint_storage: CheckpointStorageConfig = Field(default_factory=S3CheckpointStorage)
    mlflow_tracking_uri: str = settings.MLFLOW_TRACKING_URI
    experiment_name: str = "one-tower-lcm"


# ---------------------------------------------------------------------------
# DLCM model spec — token-level hierarchical model (mirrors the LCM patterns)
# ---------------------------------------------------------------------------


class PresetDLCMModelSpec(BaseModel):
    spec_type: Literal["preset"] = "preset"
    size: DLCMSize = "small"
    vocab_size: int = 50257
    max_seq_len: int = 1024
    target_ratio: float = 4.0
    boundary_mode: Literal["learned", "rule"] = "learned"


class CustomDLCMModelSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    spec_type: Literal["custom"] = "custom"
    config: DLCMConfig


DLCMModelSpec = Annotated[PresetDLCMModelSpec | CustomDLCMModelSpec, Field(discriminator="spec_type")]


@singledispatch
def resolve_dlcm_config(spec: object) -> DLCMConfig:
    raise TypeError(f"Unknown DLCM model spec type: {type(spec)}")


@resolve_dlcm_config.register(PresetDLCMModelSpec)
def _(spec: PresetDLCMModelSpec) -> DLCMConfig:
    return dlcm_config_for_size(
        spec.size,
        vocab_size=spec.vocab_size,
        max_seq_len=spec.max_seq_len,
        target_ratio=spec.target_ratio,
        boundary_mode=spec.boundary_mode,
    )


@resolve_dlcm_config.register(CustomDLCMModelSpec)
def _(spec: CustomDLCMModelSpec) -> DLCMConfig:
    return spec.config


# ---------------------------------------------------------------------------
# DLCM data / training config — packed token shards
# ---------------------------------------------------------------------------


class TokenDataConfig(BaseModel):
    tokens_dir: Path
    num_workers: int = 2


class DLCMTrainingConfig(TrainingConfig):
    aux_loss_weight: float = 0.03
    micro_batch_size: int = 4
    accumulate_grad_batches: int = 8
    precision: str = "bf16-mixed"
    val_check_interval: int = 1000
    warm_start_embedding: bool = False
    # Requires model.vocab_size == source tokenizer's vocab (token ids must
    # correspond); PCA-reduced to d_token. See domain.usecases.warm_start.
    warm_start_source: str = "gpt2"


class TrainDLCMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init: TrainingInitConfig = Field(default_factory=NewRunInit)
    data: TokenDataConfig
    model: DLCMModelSpec = Field(default_factory=PresetDLCMModelSpec)
    training: DLCMTrainingConfig = Field(default_factory=DLCMTrainingConfig)
    checkpoint_storage: CheckpointStorageConfig = Field(default_factory=S3CheckpointStorage)
    mlflow_tracking_uri: str = settings.MLFLOW_TRACKING_URI
    experiment_name: str = "dlcm"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class TrainBaseLCMResult(BaseModel):
    run_id: str
    best_val_loss: float | None
    final_step: int
