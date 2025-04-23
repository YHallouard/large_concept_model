import lightning as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from lcm_explo.domain.models import RMSELoss
from lcm_explo.domain.models.base_lcm import BaseLCM, BaseLCMConfig


class BaseLCMTrainingModule(pl.LightningModule):
    def __init__(
        self,
        config: BaseLCMConfig,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss = RMSELoss(100)

        self.model = BaseLCM(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.model(x, padding_mask)
        return output

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, padding_mask = batch

        y_pred = self(x, padding_mask)
        loss: torch.Tensor = self.loss(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, padding_mask = batch

        y_pred = self(x, padding_mask)
        loss: torch.Tensor = self.loss(y_pred, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.warmup_steps // 10, T_mult=1, eta_min=self.min_lr)

        return OptimizerLRSchedulerConfig(**{
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        })
