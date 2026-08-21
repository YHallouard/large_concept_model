import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Vaswani-style absolute sinusoidal positional encoding, added to the input."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe = torch.zeros(1, max_seq_len, d_model)  # (1, S, D)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """
        return x + self.pe[:, : x.shape[1], :]
