import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import tanh


class DyT(nn.Module):
    def __init__(self, embed_dim: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = Parameter(torch.ones(1) * init_alpha)
        self.gamma = Parameter(torch.ones(embed_dim))
        self.beta = Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tanh(self.alpha * x)
        return self.gamma * x + self.beta
