import torch
from torch.utils.data import Dataset

from lcm_explo.domain.infrastructures.token_sequences import TokenSequenceRepository


class InMemoryTokensDataset(Dataset, TokenSequenceRepository):
    """In-memory packed-token dataset (test double for ``PackedTokensDataset``).

    Concatenates a list of token documents, inserting ``eos_token_id`` between
    them, and exposes non-overlapping ``seq_len``-token windows as shifted
    ``(input_ids, labels)`` pairs.
    """

    def __init__(self, documents: list[list[int]], seq_len: int = 32, eos_token_id: int = 50256) -> None:
        self.seq_len = seq_len
        self.window = seq_len + 1

        buffer: list[int] = []
        for doc in documents:
            buffer.extend(doc)
            buffer.append(eos_token_id)
        self._tokens = torch.tensor(buffer, dtype=torch.int64)

        self._num_sequences = max(0, (len(self._tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self._num_sequences

    def get_sequence(self, idx: int) -> torch.Tensor:
        offset = idx * self.seq_len
        return self._tokens[offset : offset + self.window]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.get_sequence(idx)
        return tokens[:-1], tokens[1:]
