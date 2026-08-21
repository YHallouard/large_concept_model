from __future__ import annotations

import torch
from prefect import flow, get_run_logger

from lcm_explo.adapters.dataset.embedding._class import EmbeddingsDataset, embedding_collate_fn
from lcm_explo.adapters.dataset.packed_tokens._class import PackedTokensDataset, packed_tokens_collate_fn
from lcm_explo.domain.usecases.train.train_base_lcm import BaseLCMTrainingModule
from lcm_explo.domain.usecases.train.train_dlcm import DLCMTrainingModule
from lcm_explo.domain.usecases.train.train_one_tower_lcm import OneTowerLCMTrainingModule
from lcm_explo.domain.usecases.warm_start import warm_start_embedding_from_gpt2
from lcm_explo.utils._model_size import count_parameters, format_parameter_count
from lcm_explo.utils.checkpoints import get_checkpoint_handler
from lcm_explo.workflows._inputs import (
    DataConfig,
    NewRunInit,
    ResumeRunInit,
    TrainBaseLCMConfig,
    TrainBaseLCMResult,
    TrainDLCMConfig,
    TrainingConfig,
    TrainOneTowerLCMConfig,
    resolve_base_lcm_config,
    resolve_dlcm_config,
    resolve_one_tower_config,
    resolve_run_id,
)

try:
    import lightning as pl
    from lightning.pytorch.callbacks import LearningRateMonitor
    from lightning.pytorch.loggers import MLFlowLogger
    from torch.utils.data import DataLoader, Subset
except ImportError as e:
    raise ImportError("lightning is required for training flows") from e


def _detect_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _needs_offload(n_params: int, vram_budget_gb: float | None, accelerator: str) -> bool:
    """True si le modèle dépasse le budget VRAM sur GPU.

    Estimation conservatrice : poids fp32 + grad fp32 + 2 états Adam fp32 = 16 o/param.
    CPU et MPS ne supportent pas DeepSpeed → toujours False.
    """
    if accelerator != "gpu" or vram_budget_gb is None:
        return False
    return n_params * 16 / 1e9 > vram_budget_gb


def _build_strategy(
    n_params: int,
    vram_budget_gb: float | None,
    accelerator: str,
) -> str | object:
    """Retourne 'auto' ou DeepSpeedStrategy(ZeRO-2 + offload_optimizer) selon le budget."""
    if not _needs_offload(n_params, vram_budget_gb, accelerator):
        return "auto"
    try:
        from lightning.pytorch.strategies import DeepSpeedStrategy
    except ImportError:
        raise ImportError(
            "DeepSpeed requis pour l'offload CPU : pip install 'lcm_explo[deepspeed]'"
        ) from None
    return DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        allgather_bucket_size=5e8,
        reduce_bucket_size=5e8,
    )


def _build_loaders(
    data: DataConfig,
    training: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = EmbeddingsDataset(data.embeddings_dir, data.sequence_length, data.stride)
    n_train = int(0.9 * len(dataset.documents))
    train_docs = set(dataset.documents[:n_train])
    val_docs = set(dataset.documents[n_train:])
    train_indices = [i for i, (doc_id, _) in enumerate(dataset.sequence_indices) if doc_id in train_docs]
    val_indices = [i for i, (doc_id, _) in enumerate(dataset.sequence_indices) if doc_id in val_docs]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=training.batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=training.batch_size,
        collate_fn=embedding_collate_fn,
    )
    return train_loader, val_loader


@flow(name="train-base-lcm", log_prints=True)
def train_base_lcm_flow(config: TrainBaseLCMConfig) -> TrainBaseLCMResult:
    logger = get_run_logger()

    pl.seed_everything(config.training.seed)
    run_id = resolve_run_id(config.init)
    logger.info("Training run: %s", run_id)

    mlflow_logger = MLFlowLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
        run_name=run_id,
        log_model=False,
    )

    handler = get_checkpoint_handler(config.checkpoint_storage)
    train_loader, val_loader = _build_loaders(config.data, config.training)

    model_config = resolve_base_lcm_config(config.model)
    module = BaseLCMTrainingModule(
        config=model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.training.min_lr,
        checkpoint_handler=handler,
        run_id=run_id,
        checkpoint_every_n_steps=config.training.checkpoint_every_n_steps,
        resume_init=config.init if isinstance(config.init, ResumeRunInit) else None,
    )

    n_params = count_parameters(module.model)
    logger.info("Model: %s parameters", format_parameter_count(n_params))

    # Load frozen normalizer stats on fresh runs; resume gets them from checkpoint
    normalizer_path = config.data.embeddings_dir / "normalizer.pt"
    if isinstance(config.init, NewRunInit) and normalizer_path.exists():
        module.model.load_normalizer_stats(normalizer_path)
        logger.info("Loaded normalizer stats from %s", normalizer_path)

    mlflow_logger.log_hyperparams({"num_parameters": n_params, "model_size": format_parameter_count(n_params)})

    accelerator = _detect_accelerator()
    offloading = _needs_offload(n_params, config.training.vram_budget_gb, accelerator)
    strategy   = _build_strategy(n_params, config.training.vram_budget_gb, accelerator)
    precision  = "bf16-mixed" if (accelerator == "gpu" and offloading) else (
                 "16-mixed"   if accelerator == "gpu" else "32")
    if offloading:
        logger.info(
            "VRAM estimée %.1f GB > budget %.1f GB — DeepSpeed ZeRO-2 offload activé",
            n_params * 16 / 1e9, config.training.vram_budget_gb,
        )

    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=mlflow_logger,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(module, train_loader, val_loader)

    return TrainBaseLCMResult(
        run_id=run_id,
        best_val_loss=module.best_val_loss if module.best_val_loss < float("inf") else None,
        final_step=trainer.global_step,
    )


@flow(name="train-dlcm", log_prints=True)
def train_dlcm_flow(config: TrainDLCMConfig) -> TrainBaseLCMResult:
    logger = get_run_logger()

    pl.seed_everything(config.training.seed)
    run_id = resolve_run_id(config.init)
    logger.info("Training run: %s", run_id)

    mlflow_logger = MLFlowLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
        run_name=run_id,
        log_model=False,
    )

    handler = get_checkpoint_handler(config.checkpoint_storage)

    train_dataset = PackedTokensDataset(config.data.tokens_dir, split="train")
    val_dataset = PackedTokensDataset(config.data.tokens_dir, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=packed_tokens_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.micro_batch_size,
        num_workers=config.data.num_workers,
        collate_fn=packed_tokens_collate_fn,
    )

    model_config = resolve_dlcm_config(config.model)
    module = DLCMTrainingModule(
        config=model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.training.min_lr,
        aux_loss_weight=config.training.aux_loss_weight,
        checkpoint_handler=handler,
        run_id=run_id,
        checkpoint_every_n_steps=config.training.checkpoint_every_n_steps,
        resume_init=config.init if isinstance(config.init, ResumeRunInit) else None,
    )

    if isinstance(config.init, NewRunInit) and config.training.warm_start_embedding:
        applied = warm_start_embedding_from_gpt2(module.model, source_model_name=config.training.warm_start_source)
        logger.info(
            "Embedding warm-start from %s: %s",
            config.training.warm_start_source,
            "applied" if applied else "skipped (vocab mismatch)",
        )

    n_params = count_parameters(module.model)
    logger.info("Model: %s parameters", format_parameter_count(n_params))
    mlflow_logger.log_hyperparams({"num_parameters": n_params, "model_size": format_parameter_count(n_params)})

    accelerator = _detect_accelerator()
    precision = config.training.precision if accelerator == "gpu" else "32"

    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        logger=mlflow_logger,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(module, train_loader, val_loader)

    return TrainBaseLCMResult(
        run_id=run_id,
        best_val_loss=module.best_val_loss if module.best_val_loss < float("inf") else None,
        final_step=trainer.global_step,
    )


@flow(name="train-one-tower-lcm", log_prints=True)
def train_one_tower_lcm_flow(config: TrainOneTowerLCMConfig) -> TrainBaseLCMResult:
    logger = get_run_logger()

    pl.seed_everything(config.training.seed)
    run_id = resolve_run_id(config.init)
    logger.info("Training run: %s", run_id)

    mlflow_logger = MLFlowLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
        run_name=run_id,
        log_model=False,
    )

    handler = get_checkpoint_handler(config.checkpoint_storage)
    train_loader, val_loader = _build_loaders(config.data, config.training)

    model_config = resolve_one_tower_config(config.model)
    module = OneTowerLCMTrainingModule(
        config=model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.training.min_lr,
        checkpoint_handler=handler,
        run_id=run_id,
        checkpoint_every_n_steps=config.training.checkpoint_every_n_steps,
        resume_init=config.init if isinstance(config.init, ResumeRunInit) else None,
    )

    n_params = count_parameters(module.model)
    logger.info("Model: %s parameters", format_parameter_count(n_params))
    mlflow_logger.log_hyperparams({"num_parameters": n_params, "model_size": format_parameter_count(n_params)})

    normalizer_path = config.data.embeddings_dir / "normalizer.pt"
    if isinstance(config.init, NewRunInit) and normalizer_path.exists():
        module.model.load_normalizer_stats(normalizer_path)
        logger.info("Loaded normalizer stats from %s", normalizer_path)

    accelerator = _detect_accelerator()
    offloading = _needs_offload(n_params, config.training.vram_budget_gb, accelerator)
    strategy   = _build_strategy(n_params, config.training.vram_budget_gb, accelerator)
    precision  = "bf16-mixed" if (accelerator == "gpu" and offloading) else (
                 "16-mixed"   if accelerator == "gpu" else "32")
    if offloading:
        logger.info(
            "VRAM estimée %.1f GB > budget %.1f GB — DeepSpeed ZeRO-2 offload activé",
            n_params * 16 / 1e9, config.training.vram_budget_gb,
        )

    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=mlflow_logger,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(module, train_loader, val_loader)

    return TrainBaseLCMResult(
        run_id=run_id,
        best_val_loss=module.best_val_loss if module.best_val_loss < float("inf") else None,
        final_step=trainer.global_step,
    )
