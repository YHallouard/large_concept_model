import torch
import torch.nn as nn


def segment_ids_from_boundaries(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map boundary indicators to 0-based, non-decreasing segment ids.

    ``b`` is (B, L) with b[:, 0] == 1. Returns ``seg_id`` (B, L) int64 and
    ``num_segments`` (B,) int64 = number of concepts per sequence.
    """
    seg_id = b.long().cumsum(dim=1) - 1  # (B, L), starts at 0
    num_segments = b.long().sum(dim=1)  # (B,)
    return seg_id, num_segments


class SegmentMeanPooling(nn.Module):
    """Mean-pool token hidden states within each segment into concept vectors.

    Parameter-free: the up-projection to the concept dimension lives in the
    model so pooling stays testable in isolation. ``scatter_add_`` is
    differentiable w.r.t. ``h`` (the source), so CE gradients flow back into the
    encoder; the integer ``seg_id`` carries no gradient, keeping W_q/W_k out of
    the CE path.
    """

    def forward(
        self, h: torch.Tensor, seg_id: torch.Tensor, num_segments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, dim = h.shape
        max_segments = int(num_segments.max().item())

        index = seg_id.unsqueeze(-1).expand(-1, -1, dim)  # (B, L, d)
        sums = h.new_zeros(batch_size, max_segments, dim)
        sums.scatter_add_(1, index, h)

        counts = h.new_zeros(batch_size, max_segments)
        counts.scatter_add_(1, seg_id, torch.ones_like(seg_id, dtype=h.dtype))
        counts = counts.clamp(min=1.0)

        pooled = sums / counts.unsqueeze(-1)  # (B, M_max, d)

        arange = torch.arange(max_segments, device=h.device)
        concept_mask = arange.unsqueeze(0) < num_segments.unsqueeze(1)  # (B, M_max) bool
        return pooled, concept_mask
