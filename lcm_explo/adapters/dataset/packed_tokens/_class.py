import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from lcm_explo.domain.infrastructures.token_sequences import TokenSequenceRepository

Split = Literal["train", "val"]

META_FILENAME = "meta.json"


def packed_tokens_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels


class PackedTokensDataset(Dataset, TokenSequenceRepository):
    """Reads packed uint16 token shards written by ``prepare_token_shards_task``.

    Sequences are non-overlapping windows of ``seq_len + 1`` tokens across the
    concatenated shards. The last ``val_fraction`` of sequences form the ``val``
    split, so train/val never share windows. ``__getitem__`` returns
    ``(input_ids, labels)`` shifted by one, both int64.
    """

    def __init__(self, data_dir: Path, split: Split = "train") -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        meta = json.loads((self.data_dir / META_FILENAME).read_text())
        self.seq_len: int = meta["seq_len"]
        self.dtype = np.dtype(meta.get("dtype", "uint16"))
        total_sequences: int = meta["total_sequences"]
        val_fraction: float = meta.get("val_fraction", 0.0)

        self.window = self.seq_len + 1
        shard_names: list[str] = meta["shards"]
        self._memmaps = [
            np.memmap(self.data_dir / name, dtype=self.dtype, mode="r") for name in shard_names
        ]
        self._tokens = np.concatenate([np.asarray(mm) for mm in self._memmaps])

        num_val = int(total_sequences * val_fraction)
        num_train = total_sequences - num_val
        if split == "train":
            self._start_seq = 0
            self._num_sequences = num_train
        else:
            self._start_seq = num_train
            self._num_sequences = num_val

    def __len__(self) -> int:
        return self._num_sequences

    def get_sequence(self, idx: int) -> torch.Tensor:
        seq_index = self._start_seq + idx
        offset = seq_index * self.seq_len
        window = self._tokens[offset : offset + self.window]
        return torch.from_numpy(window.astype(np.int64))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.get_sequence(idx)
        return tokens[:-1], tokens[1:]
