from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lcm_explo.domain.models._losses import BoundaryRatioLoss
from lcm_explo.domain.models.dlcm import DLCM, DLCMConfig

if TYPE_CHECKING:
    from lcm_explo.utils.checkpoints import CheckpointHandler
    from lcm_explo.workflows._inputs import ResumeRunInit

_NUM_POSITION_BUCKETS = 16


class DLCMTrainingModule(pl.LightningModule):
    """Lightning module training a DLCM with next-token CE + load-balancing aux loss.

    Mirrors ``BaseLCMTrainingModule`` (same checkpoint hooks / ctor shape). The
    optimizer uses decoupled learning-rate groups (token vs concept width,
    ``η ∝ width^-1``) and excludes norms/biases/embeddings from weight decay.
    """

    def __init__(
        self,
        config: DLCMConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100_000,
        min_lr: float = 1e-6,
        aux_loss_weight: float = 0.03,
        lr_concept_scale: float | None = None,
        # Infrastructure — optional so tests don't need it
        checkpoint_handler: CheckpointHandler | None = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 500,
        resume_init: ResumeRunInit | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_handler", "resume_init"])

        self.model = DLCM(config)
        self.aux_loss = BoundaryRatioLoss(target_ratio=config.target_ratio)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.aux_loss_weight = aux_loss_weight
        self.lr_concept_scale = (
            lr_concept_scale if lr_concept_scale is not None else config.d_token / config.d_concept
        )

        self.checkpoint_handler = checkpoint_handler
        self.run_id = run_id
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self._resume_init = resume_init
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Lightning checkpoint hooks (identical semantics to BaseLCMTrainingModule)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        if self._resume_init is None or self.checkpoint_handler is None:
            return
        opt = self.trainer.optimizers[0]
        sched = self.trainer.lr_scheduler_configs[0].scheduler
        meta = self.checkpoint_handler.restore(self, opt, sched, self._resume_init.run_id, self._resume_init.slot)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, labels = batch
        out = self.model(input_ids)
        vocab = out.logits.size(-1)
        ce = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1))
        aux, f_rate, g = self.aux_loss(out.boundary_probs, out.boundaries)
        loss = ce + self.aux_loss_weight * aux

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)
        self.log("train_aux", aux)
        self.log("train_boundary_rate_F", f_rate)
        self.log("train_boundary_prob_G", g)
        self.log("train_compression_ratio", 1.0 / f_rate.clamp(min=1e-6), prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, labels = batch
        out = self.model(input_ids)
        vocab = out.logits.size(-1)
        ce = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1))
        self.log("val_loss", ce, prog_bar=True)

        f_rate = out.boundaries.float().mean()
        self.log("val_compression_ratio", 1.0 / f_rate.clamp(min=1e-6))
        self._log_loss_by_concept_position(out.logits, labels, out.boundaries)
        return ce

    def _log_loss_by_concept_position(
        self, logits: torch.Tensor, labels: torch.Tensor, boundaries: torch.Tensor
    ) -> None:
        """Log per-token CE bucketed by position within its concept (DLCM Fig. 7).

        A U-shaped profile (lower loss near boundaries) validates the mechanism.
        """
        vocab = logits.size(-1)
        token_ce = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none")
        token_ce = token_ce.reshape(labels.shape)  # (B, L)

        # position within concept = index minus the last boundary index at or before t
        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0).expand_as(labels)
        boundary_pos = torch.where(boundaries > 0, positions, torch.full_like(positions, -1))
        last_boundary = torch.cummax(boundary_pos, dim=1).values  # (B, L), >= 0 (pos 0 is a boundary)
        pos_in_concept = (positions - last_boundary).clamp(min=0, max=_NUM_POSITION_BUCKETS - 1)

        for bucket in range(_NUM_POSITION_BUCKETS):
            mask = pos_in_concept == bucket
            if mask.any():
                self.log(f"val_loss_pos_{bucket}", token_ce[mask].mean())

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        param_groups = self._build_param_groups()
        optimizer = AdamW(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)

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

    def _build_param_groups(self) -> list[dict]:
        """Four groups: {token, concept} x {decay, no-decay}.

        Concept-width modules (backbone, smoothing, cross-attn key/value, w_up)
        get lr * lr_concept_scale. Norms, biases and embeddings are excluded from
        weight decay.
        """
        concept_prefixes = (
            "model.backbone.",
            "model.decoder.smoothing.",
            "model.segmenter.w_up.",
        )
        concept_substrings = (
            "cross_attention.attention.k_proj",
            "cross_attention.attention.v_proj",
        )

        def is_concept(name: str) -> bool:
            return name.startswith(concept_prefixes) or any(s in name for s in concept_substrings)

        def no_decay(name: str, param: torch.Tensor) -> bool:
            return param.ndim < 2 or "embedding" in name or "norm" in name

        groups: dict[str, dict] = {
            "token_decay": {"params": [], "lr": self.learning_rate, "weight_decay": self.weight_decay},
            "token_no_decay": {"params": [], "lr": self.learning_rate, "weight_decay": 0.0},
            "concept_decay": {
                "params": [],
                "lr": self.learning_rate * self.lr_concept_scale,
                "weight_decay": self.weight_decay,
            },
            "concept_no_decay": {
                "params": [],
                "lr": self.learning_rate * self.lr_concept_scale,
                "weight_decay": 0.0,
            },
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            scope = "concept" if is_concept(name) else "token"
            decay = "no_decay" if no_decay(name, param) else "decay"
            groups[f"{scope}_{decay}"]["params"].append(param)

        return [g for g in groups.values() if g["params"]]
