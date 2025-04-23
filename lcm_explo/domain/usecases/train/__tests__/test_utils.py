import lightning as pl
from lightning import Callback


class MetricsCallback(Callback):
    """Callback to capture metrics during training and validation."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, list[float]] = {}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture metrics at the end of each training epoch."""
        for key, value in trainer.logged_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))
