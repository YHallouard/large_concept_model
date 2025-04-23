import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self, scale: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.scale = scale

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps) * self.scale
        return loss
