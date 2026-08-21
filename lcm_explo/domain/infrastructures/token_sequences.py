from abc import ABC, abstractmethod

import torch


class TokenSequenceRepository(ABC):
    """Abstract source of fixed-length packed token windows for DLCM training."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of available (non-overlapping) sequences."""

    @abstractmethod
    def get_sequence(self, idx: int) -> torch.Tensor:
        """Return one window as a 1-D int64 tensor of length ``seq_len + 1``.

        The extra token lets the training step form shifted ``(input, label)``
        pairs without crossing window boundaries.
        """
