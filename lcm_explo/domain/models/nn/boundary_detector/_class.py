from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

BoundaryMode = Literal["learned", "rule"]


class BoundaryDetector(nn.Module):
    """Detects semantic boundaries from local dissimilarity between adjacent tokens.

    boundary prob  p_t = (1 - cos(q_{t-1}, k_t)) / 2   (DLCM Eq. 6)

    In ``learned`` mode q/k are linear projections of the hidden states; in
    ``rule`` mode they are the hidden states themselves (the stable variant from
    the paper's §8.1 ablation). ``p_1`` is forced to 1 so the first token always
    starts a new concept.
    """

    def __init__(self, d_token: int, d_scan: int, mode: BoundaryMode = "learned") -> None:
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.w_q = nn.Linear(d_token, d_scan, bias=False)
            self.w_k = nn.Linear(d_token, d_scan, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "learned":
            q = self.w_q(h)
            k = self.w_k(h)
        else:
            q = k = h

        # cos(q_{t-1}, k_t) for t = 1..L-1
        cos = F.cosine_similarity(q[:, :-1], k[:, 1:], dim=-1)  # (B, L-1)
        p_rest = 0.5 * (1.0 - cos)

        batch_size = h.size(0)
        ones = h.new_ones(batch_size, 1)
        return torch.cat([ones, p_rest], dim=1)  # (B, L), p_1 == 1

    @torch.no_grad()
    def sample(self, p: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """Sharpen probabilities by temperature then draw Bernoulli boundaries.

        The returned tensor carries no gradient by design: W_q/W_k are trained
        only through the auxiliary load-balancing loss (DLCM §3.3), never CE.
        """
        eps = 1e-6
        p = p.clamp(eps, 1.0 - eps)
        inv_t = 1.0 / temperature
        p_sharp = p**inv_t / (p**inv_t + (1.0 - p) ** inv_t)
        b = torch.bernoulli(p_sharp)
        b[:, 0] = 1.0
        return b

    @torch.no_grad()
    def threshold(self, p: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """Hard-threshold boundaries for inference."""
        b = (p >= tau).to(p.dtype)
        b[:, 0] = 1.0
        return b
