import unittest

import lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lcm_explo.adapters.dataset.in_memory_tokens import InMemoryTokensDataset
from lcm_explo.adapters.dataset.packed_tokens import packed_tokens_collate_fn
from lcm_explo.domain.models.dlcm import DLCMConfig
from lcm_explo.domain.usecases.train import DLCMTrainingModule


def _tiny_config() -> DLCMConfig:
    return DLCMConfig(
        vocab_size=64,
        d_token=16,
        d_concept=32,
        d_scan=8,
        num_encoder_layers=1,
        num_backbone_layers=1,
        num_decoder_layers=1,
        num_token_heads=2,
        num_concept_heads=2,
        ffn_multiplier=2,
        max_seq_len=16,
    )


class TestDLCMTrainingModule(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _tiny_config()
        docs = [list(range(1, 40)), list(range(5, 50))]
        self.dataset = InMemoryTokensDataset(docs, seq_len=8, eos_token_id=63)

    def _module(self, **kwargs: object) -> DLCMTrainingModule:
        return DLCMTrainingModule(config=self.config, warmup_steps=10, max_steps=100, **kwargs)  # type: ignore[arg-type]

    def test_training_step_runs(self) -> None:
        # Given
        model = self._module()
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=packed_tokens_collate_fn)

        # When
        trainer = pl.Trainer(max_steps=1, logger=False, enable_checkpointing=False, enable_model_summary=False)
        trainer.fit(model, loader)

        # Then
        self.assertTrue(model.training)

    def test_validation_step_returns_scalar(self) -> None:
        # Given
        model = self._module()
        model.eval()
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=packed_tokens_collate_fn)

        # When
        batch = next(iter(loader))
        with torch.no_grad():
            val_loss = model.validation_step(batch, 0)

        # Then
        self.assertIsInstance(val_loss, torch.Tensor)
        self.assertEqual(val_loss.dim(), 0)

    def test_param_groups_split_token_and_concept(self) -> None:
        # Given
        model = self._module()

        # When
        cfg = model.configure_optimizers()
        optimizer = cfg["optimizer"]

        # Then there is a token-lr group and a scaled concept-lr group, and a
        # no-decay group with weight_decay 0. (Inspect ``initial_lr``: the warmup
        # scheduler already scaled ``lr`` down at construction.)
        lrs = {round(group["initial_lr"], 10) for group in optimizer.param_groups}
        expected_concept_lr = round(model.learning_rate * model.lr_concept_scale, 10)
        self.assertIn(round(model.learning_rate, 10), lrs)
        self.assertIn(expected_concept_lr, lrs)
        self.assertTrue(any(group["weight_decay"] == 0.0 for group in optimizer.param_groups))

    def test_all_trainable_params_covered_once(self) -> None:
        # Given
        model = self._module()

        # When
        cfg = model.configure_optimizers()
        grouped = sum(len(group["params"]) for group in cfg["optimizer"].param_groups)

        # Then every trainable parameter is assigned to exactly one group
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertEqual(grouped, trainable)

    def test_overfit_single_batch(self) -> None:
        # Given a fixed batch and a high LR
        torch.manual_seed(0)
        model = self._module(learning_rate=3e-3, aux_loss_weight=0.0)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 8))
        labels = torch.randint(0, self.config.vocab_size, (2, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        # When we overfit for a couple hundred steps
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            out = model.model(input_ids)
            ce = F.cross_entropy(out.logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            ce.backward()
            optimizer.step()

        # Then CE drops well below 1.0 (no label leakage, model can learn)
        self.assertLess(ce.item(), 1.0)
