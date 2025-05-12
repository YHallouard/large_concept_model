import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self, scale: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.eps = eps
        self.scale = scale

    def forward(self, yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(yhat, y).mean(dim=-1)  # (batch, seq, model_dim sonar)
        masked_mse_loss = mse_loss * mask

        mean_masked_mse_loss = masked_mse_loss.sum() / mask.sum()

        loss = torch.sqrt(mean_masked_mse_loss + self.eps) * self.scale
        return loss
