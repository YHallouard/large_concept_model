import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from lcm_explo.adapters.dataset.packed_tokens._class import META_FILENAME
from lcm_explo.domain.models.dlcm import DLCMConfig
from lcm_explo.utils.checkpoints import LocalCheckpointStorage
from lcm_explo.workflows._flows import train_dlcm_flow
from lcm_explo.workflows._inputs import (
    CustomDLCMModelSpec,
    DLCMTrainingConfig,
    TokenDataConfig,
    TrainDLCMConfig,
)


def _fake_gpt2(vocab_size: int, hidden_size: int) -> MagicMock:
    fake = MagicMock()
    fake.wte.weight.data = torch.randn(vocab_size, hidden_size)
    return fake


def _write_tiny_shard(data_dir: Path, seq_len: int, vocab_size: int) -> None:
    num_tokens = seq_len * 40 + 1
    tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
    tokens.tofile(data_dir / "shard_0000.bin")
    meta = {
        "seq_len": seq_len,
        "dtype": "uint16",
        "total_sequences": (num_tokens - 1) // seq_len,
        "val_fraction": 0.2,
        "shards": ["shard_0000.bin"],
    }
    (data_dir / META_FILENAME).write_text(json.dumps(meta))


def _build_config(root: Path, tokens_dir: Path, vocab_size: int, seq_len: int, **training_overrides: object) -> TrainDLCMConfig:
    return TrainDLCMConfig(
        data=TokenDataConfig(tokens_dir=tokens_dir, num_workers=0),
        model=CustomDLCMModelSpec(
            config=DLCMConfig(
                vocab_size=vocab_size,
                d_token=16,
                d_concept=32,
                d_scan=8,
                num_encoder_layers=1,
                num_backbone_layers=1,
                num_decoder_layers=1,
                num_token_heads=2,
                num_concept_heads=2,
                ffn_multiplier=2,
                max_seq_len=seq_len,
            )
        ),
        training=DLCMTrainingConfig(
            max_steps=2,
            warmup_steps=1,
            micro_batch_size=2,
            accumulate_grad_batches=1,
            val_check_interval=1,
            **training_overrides,
        ),
        checkpoint_storage=LocalCheckpointStorage(base_path=root / "ckpt"),
        mlflow_tracking_uri=f"file://{root / 'mlruns'}",
    )


class TestTrainDLCMFlow(unittest.TestCase):
    def test_flow_runs_end_to_end_on_cpu(self) -> None:
        # Given a tiny packed-token shard and a tiny custom DLCM config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokens_dir = root / "tokens"
            tokens_dir.mkdir()
            vocab_size = 64
            seq_len = 16
            _write_tiny_shard(tokens_dir, seq_len, vocab_size)
            config = _build_config(root, tokens_dir, vocab_size, seq_len)

            # When
            result = train_dlcm_flow(config)

            # Then the flow completes and reports progress
            self.assertEqual(result.final_step, 2)
            self.assertIsInstance(result.run_id, str)

    @patch("transformers.GPT2Model.from_pretrained")
    def test_flow_applies_warm_start_when_vocab_matches(self, mock_from_pretrained: MagicMock) -> None:
        # Given a mocked GPT-2 source whose vocab matches the tiny model's
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokens_dir = root / "tokens"
            tokens_dir.mkdir()
            vocab_size = 64
            seq_len = 16
            _write_tiny_shard(tokens_dir, seq_len, vocab_size)
            mock_from_pretrained.return_value = _fake_gpt2(vocab_size, hidden_size=32)
            config = _build_config(root, tokens_dir, vocab_size, seq_len, warm_start_embedding=True)

            # When
            result = train_dlcm_flow(config)

            # Then the flow completes and the (mocked) source was consulted
            self.assertEqual(result.final_step, 2)
            mock_from_pretrained.assert_called_once()

    @patch("transformers.GPT2Model.from_pretrained")
    def test_flow_skips_warm_start_on_vocab_mismatch(self, mock_from_pretrained: MagicMock) -> None:
        # Given a mocked GPT-2 source whose vocab does NOT match the tiny model's
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokens_dir = root / "tokens"
            tokens_dir.mkdir()
            vocab_size = 64
            seq_len = 16
            _write_tiny_shard(tokens_dir, seq_len, vocab_size)
            mock_from_pretrained.return_value = _fake_gpt2(vocab_size=999, hidden_size=32)
            config = _build_config(root, tokens_dir, vocab_size, seq_len, warm_start_embedding=True)

            # When — the flow still completes; warm start is skipped, not fatal
            result = train_dlcm_flow(config)

            # Then
            self.assertEqual(result.final_step, 2)
            mock_from_pretrained.assert_called_once()
