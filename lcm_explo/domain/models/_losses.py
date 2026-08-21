import torch
from torch import nn


class BoundaryRatioLoss(nn.Module):
    """Global load-balancing loss that steers compression to a target ratio R.

    DLCM Eq. 10 / §3.3.3 ("Global Parser"):

        L_aux = R/(R-1) * [ (R-1)*F*G + (1-F)*(1-G) ] - 1

    with G = mean(p) the expected boundary rate and F = mean(b) the actual one.
    The minimum (0) is reached at F = G = 1/R.

    The statistics are meant to be *global* across the batch. Under gradient
    accumulation (or DDP), a single micro-batch only sees a fraction of the
    tokens. Since the loss is linear in F and F enters detached (no gradient), we
    keep an EMA of F across micro-batches and combine it with the *local* G,
    which is exactly the gradient of the loss at the global F. G stays local so
    it remains the only gradient path into the boundary projections.
    """

    def __init__(self, target_ratio: float = 4.0, ema_momentum: float = 0.99, warmup_updates: int = 50) -> None:
        super().__init__()
        self.target_ratio = target_ratio
        self.ema_momentum = ema_momentum
        self.warmup_updates = warmup_updates
        # Persistent so it survives checkpoint/resume.
        self.register_buffer("f_ema", torch.tensor(1.0 / target_ratio))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    def forward(self, p: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_local = b.detach().to(p.dtype).mean()
        g = p.mean()  # gradient path to W_q/W_k

        if self.training:
            # DDP: all_reduce f_local across ranks here before the EMA update.
            self.f_ema.mul_(self.ema_momentum).add_(f_local * (1.0 - self.ema_momentum))
            self.num_updates.add_(1)
            f_used = self.f_ema if int(self.num_updates.item()) >= self.warmup_updates else f_local
        else:
            f_used = f_local

        r = self.target_ratio
        loss = r / (r - 1.0) * ((r - 1.0) * f_used * g + (1.0 - f_used) * (1.0 - g)) - 1.0
        return loss, f_local, g


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
