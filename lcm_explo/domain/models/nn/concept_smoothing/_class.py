import torch
import torch.nn as nn


class CausalConceptSmoothing(nn.Module):
    """Integrate adjacent concepts with a *causal* depthwise conv (DLCM §3.5.1).

    Causality over the concept axis is essential: a symmetric kernel would mix
    ``z_{k+1}`` into ``z̃_k``, and since a token attends to concepts formed up to
    its own segment, that would leak future tokens (the next concept pools them)
    into the current prediction. Left-padding keeps ``z̃_k`` a function of
    segments ``<= k`` only.
    """

    def __init__(self, d_concept: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            d_concept,
            d_concept,
            kernel_size=kernel_size,
            groups=d_concept,  # depthwise
            bias=False,
        )
        self.proj = nn.Linear(d_concept, d_concept)
        self.norm = nn.RMSNorm(d_concept)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, M, d_concept)
        x = z.transpose(1, 2)  # (B, d, M)
        x = nn.functional.pad(x, (self.left_pad, 0))  # left pad only -> causal
        x = self.conv(x)  # (B, d, M)
        x = x.transpose(1, 2)  # (B, M, d)
        return self.norm(z + self.proj(x))
