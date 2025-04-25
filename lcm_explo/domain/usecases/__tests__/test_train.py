import unittest

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from lcm_explo.adapters.dataset.in_memory import InMemoryEmbeddingsDataset
from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.models.documents import DocumentEmbeddings
from lcm_explo.domain.usecases.train import BaseLCMTrainingModule


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


class TestLCMTrainingModule(unittest.TestCase):
    def setUp(self) -> None:
        # Given
        self.config = BaseLCMConfig(
            hidden_size=8,
            max_seq_len=2,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=32,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=16,
        )

        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.max_steps = 1000

        self.doc1_embeddings = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ])
        self.doc2_embeddings = torch.tensor([
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        ])

        self.documents: list[DocumentEmbeddings] = [
            DocumentEmbeddings(document_id="doc1", embeddings=self.doc1_embeddings),
            DocumentEmbeddings(document_id="doc2", embeddings=self.doc2_embeddings),
        ]

    def test_training_step(self) -> None:
        # Given
        model = BaseLCMTrainingModule(
            config=self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

        dataset = InMemoryEmbeddingsDataset(documents=self.documents, sequence_length=3, stride=1)
        dataloader = DataLoader(dataset, batch_size=2)

        # When
        trainer = pl.Trainer(max_steps=1, logger=False, enable_checkpointing=False, enable_model_summary=False)
        trainer.fit(model, dataloader)

        # Then
        self.assertTrue(model.training)

    def test_validation_step(self) -> None:
        # Given
        model = BaseLCMTrainingModule(
            config=self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

        dataset = InMemoryEmbeddingsDataset(documents=self.documents, sequence_length=3, stride=1)
        dataloader = DataLoader(dataset, batch_size=2)

        # When
        model.eval()
        batch = next(iter(dataloader))
        with torch.no_grad():
            val_loss = model.validation_step(batch, 0)

        # Then
        self.assertIsInstance(val_loss, torch.Tensor)
        self.assertEqual(val_loss.dim(), 0)

    def test_configure_optimizers(self) -> None:
        # Given
        model = BaseLCMTrainingModule(
            config=self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

        # When
        optimizer_config = model.configure_optimizers()

        # Then
        self.assertIn("optimizer", optimizer_config)
        self.assertIn("lr_scheduler", optimizer_config)
        self.assertEqual(optimizer_config["optimizer"].param_groups[0]["lr"], self.learning_rate)
        self.assertEqual(optimizer_config["optimizer"].param_groups[0]["weight_decay"], self.weight_decay)

    def test_trainer_logging(self) -> None:
        # Given
        model = BaseLCMTrainingModule(
            config=self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

        dataset = InMemoryEmbeddingsDataset(documents=self.documents, sequence_length=3, stride=1)
        train_dataloader = DataLoader(dataset, batch_size=2)
        val_dataloader = DataLoader(dataset, batch_size=2)

        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(
            max_steps=2,
            val_check_interval=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[metrics_callback],
        )

        # When
        trainer.fit(model, train_dataloader, val_dataloader)

        # Then
        self.assertIn("train_loss", metrics_callback.metrics)
        self.assertEqual(len(metrics_callback.metrics["train_loss"]), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in metrics_callback.metrics["train_loss"]))
        self.assertTrue(all(loss >= 0 for loss in metrics_callback.metrics["train_loss"]))

        self.assertIn("val_loss", metrics_callback.metrics)
        self.assertEqual(len(metrics_callback.metrics["val_loss"]), 2)  # Two validation steps
        self.assertTrue(all(isinstance(loss, float) for loss in metrics_callback.metrics["val_loss"]))
        self.assertTrue(all(loss >= 0 for loss in metrics_callback.metrics["val_loss"]))
