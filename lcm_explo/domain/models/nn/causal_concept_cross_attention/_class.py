import torch
import torch.nn as nn

from lcm_explo.domain.models.nn.qk_rmsnorm_attention._class import QKRMSNormSDPAAttention


class CausalConceptCrossAttention(nn.Module):
    """Decoder cross-attention where token t attends only to *completed* concepts.

    Token t must see concepts of segments finished strictly before its own
    (DLCM §3.5.2): the set ``{c_0, ..., c_{seg_id[t]-1}}``. Its own segment's
    concept is a mean over the whole segment — including tokens > t — so
    attending to it would leak the label.

    We realise the ragged LxM attention as a hardware-friendly LxL causal one
    via *shifted concept replication* (§4.1). Build a key/value bank where index
    k holds ``c_{k-1}`` (index 0 = a learned BOS concept), then gather it per
    token position:

        kv_bank[b, k] = c_{k-1}          (kv_bank[b, 0] = BOS)
        z_rep[b, t]   = kv_bank[b, seg_id[t]] = c_{seg_id[t]-1}

    Under a plain causal mask, query t reaches key positions s <= t, each holding
    ``c_{seg_id[s]-1}``. Since seg_id is non-decreasing and contiguous, the
    reachable set is exactly ``{BOS, c_0, ..., c_{seg_id[t]-1}}`` — all completed
    segments, never the one still forming. Replicating a concept across the token
    span it covers weights it by that span (analogous to GQA), which is the
    intended semantics. BOS also guarantees no fully-masked row (first-segment
    tokens attend to BOS), avoiding SDPA NaNs.
    """

    def __init__(self, d_token: int, d_concept: int, num_heads: int, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = QKRMSNormSDPAAttention(
            q_dim=d_token, kv_dim=d_concept, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )
        self.bos_concept = nn.Parameter(torch.zeros(1, 1, d_concept))
        nn.init.normal_(self.bos_concept, std=0.02)

    def _replicate(self, concepts: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        batch_size, _, d_concept = concepts.shape
        bos = self.bos_concept.expand(batch_size, 1, d_concept)
        kv_bank = torch.cat([bos, concepts[:, :-1]], dim=1)  # index k -> c_{k-1}
        index = seg_id.unsqueeze(-1).expand(-1, -1, d_concept)
        return kv_bank.gather(1, index)  # (B, L, d_concept)

    def forward(self, h: torch.Tensor, concepts: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        # h: (B, L, d_token); concepts: (B, M_max, d_concept); seg_id: (B, L)
        z_rep = self._replicate(concepts, seg_id)
        attended = self.attention(query=h, key=z_rep, value=z_rep, is_causal=True)
        return attended + h
