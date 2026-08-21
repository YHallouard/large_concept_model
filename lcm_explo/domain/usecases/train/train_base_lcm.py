from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lcm_explo.domain.models import RMSELoss
from lcm_explo.domain.models.base_lcm import BaseLCM, BaseLCMConfig

if TYPE_CHECKING:
    from lcm_explo.utils.checkpoints import CheckpointHandler
    from lcm_explo.workflows._inputs import ResumeRunInit


class BaseLCMTrainingModule(pl.LightningModule):
    def __init__(
        self,
        config: BaseLCMConfig,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100_000,
        min_lr: float = 1e-6,
        # Infrastructure — optional so existing tests don't break
        checkpoint_handler: CheckpointHandler | None = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 500,
        resume_init: ResumeRunInit | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_handler", "resume_init"])

        self.loss = RMSELoss(100)
        self.model = BaseLCM(config)

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
        """Restore checkpoint after configure_optimizers() has been called."""
        if self._resume_init is None or self.checkpoint_handler is None:
            return
        opt = self.trainer.optimizers[0]
        sched = self.trainer.lr_scheduler_configs[0].scheduler
        meta = self.checkpoint_handler.restore(self, opt, sched, self._resume_init.run_id, self._resume_init.slot)
        self.best_val_loss = meta.val_loss if meta.val_loss is not None else float("inf")
        self._resume_init = None  # prevent double-restore

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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.model(x, padding_mask)
        return output

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, padding_mask = batch
        y_pred = self(x, padding_mask)                            # raw SONAR space
        loss: torch.Tensor = self.loss(y_pred, y, padding_mask)  # MSE in SONAR space (paper eq. 6)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, padding_mask = batch
        y_pred = self(x, padding_mask)                            # raw SONAR space
        loss: torch.Tensor = self.loss(y_pred, y, padding_mask)
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
