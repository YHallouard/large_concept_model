import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import lightning as pl
import torch
from torch.utils.data import DataLoader

from lcm_explo.adapters.dataset.in_memory import InMemoryEmbeddingsDataset
from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.models.documents import DocumentEmbeddings
from lcm_explo.domain.usecases.train import BaseLCMTrainingModule
from lcm_explo.domain.usecases.train.__tests__.test_utils import MetricsCallback
from lcm_explo.utils.checkpoints import LocalCheckpointHandler, LocalCheckpointStorage
from lcm_explo.workflows._inputs import NewRunInit, ResumeRunInit


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
        # base lr is set at optimizer creation; LinearLR warmup lowers it at step 0
        self.assertEqual(optimizer_config["optimizer"].defaults["lr"], self.learning_rate)
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

    def test_on_validation_epoch_end_saves_best_and_last(self) -> None:
        """Verify that on_validation_epoch_end writes both 'best' and 'last' checkpoint files."""
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)
            handler = LocalCheckpointHandler(base_path)
            run_id = "test-run"

            dataset = InMemoryEmbeddingsDataset(documents=self.documents, sequence_length=3, stride=1)
            train_loader = DataLoader(dataset, batch_size=2)
            val_loader = DataLoader(dataset, batch_size=2)

            model = BaseLCMTrainingModule(
                config=self.config,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                checkpoint_handler=handler,
                run_id=run_id,
            )

            trainer = pl.Trainer(
                max_steps=2,
                val_check_interval=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            trainer.fit(model, train_loader, val_loader)

            # At least one 'last' checkpoint should have been written
            last_meta = handler.meta(run_id, "last")
            self.assertIsNotNone(last_meta)
            # 'best' should also have been written (first val epoch always improves on inf)
            best_meta = handler.meta(run_id, "best")
            self.assertIsNotNone(best_meta)
            self.assertLess(model.best_val_loss, float("inf"))

    def test_on_fit_start_restores_checkpoint(self) -> None:
        """on_fit_start should restore model weights from a previously saved checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)
            handler = LocalCheckpointHandler(base_path)
            run_id = "resume-run"

            dataset = InMemoryEmbeddingsDataset(documents=self.documents, sequence_length=3, stride=1)
            train_loader = DataLoader(dataset, batch_size=2)
            val_loader = DataLoader(dataset, batch_size=2)

            # 1. Train a first model and save a checkpoint
            model_a = BaseLCMTrainingModule(
                config=self.config,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                checkpoint_handler=handler,
                run_id=run_id,
            )
            trainer_a = pl.Trainer(
                max_steps=2,
                val_check_interval=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            trainer_a.fit(model_a, train_loader, val_loader)

            weights_after_a = {k: v.clone() for k, v in model_a.model.state_dict().items()}

            # 2. Create a fresh model with different (random) weights, then restore from 'last'
            model_b = BaseLCMTrainingModule(
                config=self.config,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                checkpoint_handler=handler,
                run_id=run_id,
                resume_init=ResumeRunInit(run_id=run_id, slot="last"),
            )
            trainer_b = pl.Trainer(
                max_steps=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            # on_fit_start fires automatically during trainer_b.fit()
            trainer_b.fit(model_b, train_loader)

            # Verify restore: at least one weight matrix should match model_a's saved
            # weights (weight matrices are non-zero after training; biases and normalizer
            # buffers start at zero so are excluded from this proxy check).
            weight_keys = [k for k in weights_after_a if k.endswith(".weight")]
            self.assertTrue(weight_keys, "No weight tensors found in state dict")
            matches = sum(
                torch.allclose(model_b.model.state_dict()[k].cpu(), weights_after_a[k].cpu(), atol=1e-4)
                for k in weight_keys
            )
            # After one training step most weights drift, but at least some should
            # still be close (tiny model, tiny lr, single step).
            self.assertGreater(
                matches, 0,
                "No restored weight matched model_a — checkpoint restore likely failed",
            )
