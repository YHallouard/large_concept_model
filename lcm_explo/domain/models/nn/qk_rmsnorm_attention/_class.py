import torch
import torch.nn as nn
import torch.nn.functional as F


class QKRMSNormSDPAAttention(nn.Module):
    """Multi-head attention with per-head RMSNorm on Q/K, backed by SDPA.

    Supports heterogeneous query/key-value dimensions (DLCM cross-attention:
    queries live in the token space, keys/values in the concept space).
    The output is projected back to the query dimension.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.k_proj = nn.Linear(kv_dim, inner_dim)
        self.v_proj = nn.Linear(kv_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, q_dim)
        # Norm applied per-head (over head_dim) after the Q/K reshape
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, target_length, _ = query.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, target_length, -1)

        return self.out_proj(attention_output)
