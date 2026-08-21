from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lcm_explo.domain.models.one_tower_lcm import OneTowerLCM, OneTowerLCMConfig

if TYPE_CHECKING:
    from lcm_explo.utils.checkpoints import CheckpointHandler
    from lcm_explo.workflows._inputs import ResumeRunInit


class OneTowerLCMTrainingModule(pl.LightningModule):
    def __init__(
        self,
        config: OneTowerLCMConfig,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100_000,
        min_lr: float = 1e-6,
        checkpoint_handler: CheckpointHandler | None = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 500,
        resume_init: ResumeRunInit | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_handler", "resume_init"])

        self.model = OneTowerLCM(config)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr

        self.checkpoint_handler = checkpoint_handler
        self.run_id = run_id
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self._resume_init = resume_init
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        if self._resume_init is None or self.checkpoint_handler is None:
            return
        opt = self.trainer.optimizers[0]
        sched = self.trainer.lr_scheduler_configs[0].scheduler
        meta = self.checkpoint_handler.restore(self, opt, sched, self._resume_init.run_id, self._resume_init.slot)  # type: ignore[arg-type]
        self.best_val_loss = meta.val_loss if meta.val_loss is not None else float("inf")
        self._resume_init = None

    def on_train_batch_end(self, outputs: object, batch: object, batch_idx: int) -> None:
        if (
            self.checkpoint_handler
            and self.run_id
            and self.global_step > 0
            and self.global_step % self.checkpoint_every_n_steps == 0
        ):
            opt = self.trainer.optimizers[0]
            sched = self.trainer.lr_scheduler_configs[0].scheduler
            self.checkpoint_handler.save(self, opt, sched, "last", self.run_id, self.global_step, None)

    def on_validation_epoch_end(self) -> None:
        val_loss_tensor = self.trainer.callback_metrics.get("val_loss")
        if val_loss_tensor is None or not self.checkpoint_handler or not self.run_id:
            return
        loss = float(val_loss_tensor)
        opt = self.trainer.optimizers[0]
        sched = self.trainer.lr_scheduler_configs[0].scheduler
        self.checkpoint_handler.save(self, opt, sched, "last", self.run_id, self.global_step, loss)
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.checkpoint_handler.save(self, opt, sched, "best", self.run_id, self.global_step, loss)

    # ------------------------------------------------------------------
    # Forward / steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, _padding_mask = batch
        drop = torch.rand(1).item() < self.model.config.cfg_prob
        outputs = self.model(x, y, drop_attn=drop)
        loss: torch.Tensor = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, _padding_mask = batch
        outputs = self.model(x, y, drop_attn=False)  # always conditional in validation
        loss: torch.Tensor = outputs["loss"]
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8 / self.learning_rate,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_steps - self.warmup_steps),
            eta_min=self.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        return OptimizerLRSchedulerConfig(**{
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        })
