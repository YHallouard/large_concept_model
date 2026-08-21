import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lcm_explo.adapters.dataset.in_memory_tokens._class import InMemoryTokensDataset
from lcm_explo.adapters.dataset.packed_tokens._class import (
    PackedTokensDataset,
    packed_tokens_collate_fn,
)


class TestPackedTokensDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.seq_len = 4
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        # 21 tokens -> (21 - 1) // 4 = 5 sequences
        self.tokens = np.arange(21, dtype=np.uint16)
        self.tokens.tofile(self.data_dir / "shard_0000.bin")
        meta = {
            "seq_len": self.seq_len,
            "dtype": "uint16",
            "total_sequences": 5,
            "val_fraction": 0.2,  # 1 val sequence, 4 train
            "shards": ["shard_0000.bin"],
        }
        (self.data_dir / "meta.json").write_text(json.dumps(meta))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_labels_are_inputs_shifted_by_one(self) -> None:
        # Given
        dataset = PackedTokensDataset(self.data_dir, split="train")

        # When
        input_ids, labels = dataset[0]

        # Then labels are inputs shifted by one, first window = tokens 0..4
        self.assertEqual(input_ids.dtype, torch.int64)
        torch.testing.assert_close(input_ids, torch.arange(0, 4, dtype=torch.int64))
        torch.testing.assert_close(labels, torch.arange(1, 5, dtype=torch.int64))

    def test_windows_are_non_overlapping_across_index(self) -> None:
        # Given
        dataset = PackedTokensDataset(self.data_dir, split="train")

        # When
        first, _ = dataset[0]
        second, _ = dataset[1]

        # Then window 1 starts one seq_len further in
        torch.testing.assert_close(second, torch.arange(4, 8, dtype=torch.int64))

    def test_train_val_split_sizes(self) -> None:
        # Given
        train = PackedTokensDataset(self.data_dir, split="train")
        val = PackedTokensDataset(self.data_dir, split="val")

        # Then val takes the last fraction, train the rest
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 1)

    def test_val_reads_from_the_tail(self) -> None:
        # Given the val split (last sequence, index 4 -> offset 16)
        val = PackedTokensDataset(self.data_dir, split="val")

        # When
        input_ids, _ = val[0]

        # Then
        torch.testing.assert_close(input_ids, torch.arange(16, 20, dtype=torch.int64))

    def test_collate_stacks_batch(self) -> None:
        # Given
        dataset = PackedTokensDataset(self.data_dir, split="train")

        # When
        input_ids, labels = packed_tokens_collate_fn([dataset[0], dataset[1]])

        # Then
        self.assertEqual(input_ids.shape, (2, self.seq_len))
        self.assertEqual(labels.shape, (2, self.seq_len))

    def test_in_memory_double_matches_contract(self) -> None:
        # Given documents packed with an eos separator
        dataset = InMemoryTokensDataset([[1, 2, 3], [4, 5, 6]], seq_len=3, eos_token_id=99)

        # When
        input_ids, labels = dataset[0]

        # Then shapes/contract match PackedTokensDataset
        self.assertEqual(input_ids.shape, (3,))
        self.assertEqual(labels.shape, (3,))
        torch.testing.assert_close(labels, dataset.get_sequence(0)[1:])
