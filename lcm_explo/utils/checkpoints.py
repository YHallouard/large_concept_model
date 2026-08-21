from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from lcm_explo.utils._minio import get_minio_client
from lcm_explo.utils._settings import settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from lcm_explo.domain.usecases.train.train_base_lcm import BaseLCMTrainingModule

CheckpointSlot = Literal["best", "last"]


# ---------------------------------------------------------------------------
# Storage config (discriminated union)
# ---------------------------------------------------------------------------


class LocalCheckpointStorage(BaseModel):
    storage: Literal["local"] = "local"
    base_path: Path


class S3CheckpointStorage(BaseModel):
    storage: Literal["s3"] = "s3"
    bucket: str = settings.MINIO_BUCKET


CheckpointStorageConfig = LocalCheckpointStorage | S3CheckpointStorage


class CheckpointMeta(BaseModel):
    step: int
    val_loss: float | None
    run_id: str


# ---------------------------------------------------------------------------
# Handler ABC
# ---------------------------------------------------------------------------


class CheckpointHandler(ABC):
    @abstractmethod
    def save(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        slot: CheckpointSlot,
        run_id: str,
        step: int,
        val_loss: float | None,
    ) -> None: ...

    @abstractmethod
    def restore(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        run_id: str,
        slot: CheckpointSlot,
    ) -> CheckpointMeta: ...

    @abstractmethod
    def meta(self, run_id: str, slot: CheckpointSlot) -> CheckpointMeta | None: ...


# ---------------------------------------------------------------------------
# Shared serialization helpers
# ---------------------------------------------------------------------------


def _serialize_module(module: BaseLCMTrainingModule) -> bytes:
    """Returns model safetensors bytes. Normalizer mean/std are in the state_dict as buffers.

    Tensors that share storage (e.g. DLCM's tied embedding / LM head) are stored
    once — safetensors rejects duplicate storages. Untied models are unaffected.
    """
    state_dict = module.model.state_dict()
    seen: dict[int, str] = {}
    deduped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        ptr = value.data_ptr()
        if ptr in seen:
            continue  # tied duplicate — restored via shared storage on load
        seen[ptr] = key
        deduped[key] = value.cpu()
    return safetensors_save(deduped)


def _restore_module(module: BaseLCMTrainingModule, model_bytes: bytes) -> None:
    state_dict = safetensors_load(model_bytes)
    # Tied params are absent from the deduped checkpoint; they share storage with a
    # loaded key so are updated in place. strict=False tolerates only those.
    missing, unexpected = module.model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    live = module.model.state_dict()
    loaded_ptrs = {live[k].data_ptr() for k in state_dict if k in live}
    untied_missing = [k for k in missing if live[k].data_ptr() not in loaded_ptrs]
    if untied_missing:
        raise RuntimeError(f"Missing keys in checkpoint: {untied_missing}")


def _serialize_optimizer(optimizer: torch.optim.Optimizer) -> bytes:
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    return buf.getvalue()


def _restore_optimizer(optimizer: torch.optim.Optimizer, data: bytes) -> None:
    optimizer.load_state_dict(torch.load(io.BytesIO(data)))  # nosec


def _serialize_scheduler(scheduler: object) -> bytes:
    buf = io.BytesIO()
    torch.save(scheduler.state_dict(), buf)  # type: ignore[attr-defined]
    return buf.getvalue()


def _restore_scheduler(scheduler: object, data: bytes) -> None:
    scheduler.load_state_dict(torch.load(io.BytesIO(data)))  # type: ignore[attr-defined]  # nosec


# ---------------------------------------------------------------------------
# S3 implementation
# ---------------------------------------------------------------------------


def _s3_upload(s3: S3Client, data: bytes, key: str, bucket: str) -> None:
    s3.upload_fileobj(io.BytesIO(data), bucket, key)


def _s3_download(s3: S3Client, key: str, bucket: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return buf.getvalue()


class S3CheckpointHandler(CheckpointHandler):
    def __init__(self, bucket: str = settings.MINIO_BUCKET) -> None:
        self._s3: S3Client = get_minio_client()
        self._bucket = bucket

    def _prefix(self, run_id: str, slot: CheckpointSlot) -> str:
        return f"lcm/{run_id}/{slot}"

    def save(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        slot: CheckpointSlot,
        run_id: str,
        step: int,
        val_loss: float | None,
    ) -> None:
        prefix = self._prefix(run_id, slot)
        _s3_upload(self._s3, _serialize_module(module), f"{prefix}/model.safetensors", self._bucket)
        _s3_upload(self._s3, _serialize_optimizer(optimizer), f"{prefix}/optimizer.pt", self._bucket)
        _s3_upload(self._s3, _serialize_scheduler(scheduler), f"{prefix}/scheduler.pt", self._bucket)
        meta = CheckpointMeta(step=step, val_loss=val_loss, run_id=run_id)
        _s3_upload(self._s3, meta.model_dump_json().encode(), f"{prefix}/meta.json", self._bucket)

    def restore(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        run_id: str,
        slot: CheckpointSlot,
    ) -> CheckpointMeta:
        prefix = self._prefix(run_id, slot)
        model_bytes = _s3_download(self._s3, f"{prefix}/model.safetensors", self._bucket)
        opt_bytes = _s3_download(self._s3, f"{prefix}/optimizer.pt", self._bucket)
        sched_bytes = _s3_download(self._s3, f"{prefix}/scheduler.pt", self._bucket)
        meta_bytes = _s3_download(self._s3, f"{prefix}/meta.json", self._bucket)
        _restore_module(module, model_bytes)
        _restore_optimizer(optimizer, opt_bytes)
        _restore_scheduler(scheduler, sched_bytes)
        return CheckpointMeta.model_validate_json(meta_bytes)

    def meta(self, run_id: str, slot: CheckpointSlot) -> CheckpointMeta | None:
        prefix = self._prefix(run_id, slot)
        try:
            meta_bytes = _s3_download(self._s3, f"{prefix}/meta.json", self._bucket)
            return CheckpointMeta.model_validate_json(meta_bytes)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------


class LocalCheckpointHandler(CheckpointHandler):
    def __init__(self, base_path: Path) -> None:
        self._base = base_path

    def _dir(self, run_id: str, slot: CheckpointSlot) -> Path:
        d = self._base / "lcm" / run_id / slot
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        slot: CheckpointSlot,
        run_id: str,
        step: int,
        val_loss: float | None,
    ) -> None:
        d = self._dir(run_id, slot)
        (d / "model.safetensors").write_bytes(_serialize_module(module))
        (d / "optimizer.pt").write_bytes(_serialize_optimizer(optimizer))
        (d / "scheduler.pt").write_bytes(_serialize_scheduler(scheduler))
        meta = CheckpointMeta(step=step, val_loss=val_loss, run_id=run_id)
        (d / "meta.json").write_text(meta.model_dump_json())

    def restore(
        self,
        module: BaseLCMTrainingModule,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        run_id: str,
        slot: CheckpointSlot,
    ) -> CheckpointMeta:
        d = self._dir(run_id, slot)
        _restore_module(module, (d / "model.safetensors").read_bytes())
        _restore_optimizer(optimizer, (d / "optimizer.pt").read_bytes())
        _restore_scheduler(scheduler, (d / "scheduler.pt").read_bytes())
        return CheckpointMeta.model_validate_json((d / "meta.json").read_text())

    def meta(self, run_id: str, slot: CheckpointSlot) -> CheckpointMeta | None:
        meta_path = self._dir(run_id, slot) / "meta.json"
        if not meta_path.exists():
            return None
        return CheckpointMeta.model_validate_json(meta_path.read_text())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_checkpoint_handler(config: CheckpointStorageConfig) -> CheckpointHandler:
    if isinstance(config, S3CheckpointStorage):
        return S3CheckpointHandler(bucket=config.bucket)
    return LocalCheckpointHandler(base_path=config.base_path)
