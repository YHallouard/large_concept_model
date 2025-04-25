import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        self.rotation_matrix = torch.zeros(d_model, d_model)
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(torch.tensor(i * j * 0.01))

        self.positional_embedding = torch.zeros(max_seq_len, d_model)
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(torch.tensor(i * j * 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """
        self.positional_embedding = self.positional_embedding.to(x.device)
        self.rotation_matrix = self.rotation_matrix.to(x.device)

        x += self.positional_embedding
        x = torch.matmul(x, self.rotation_matrix)

        return x
