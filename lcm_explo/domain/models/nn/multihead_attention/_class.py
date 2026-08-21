import torch
import torch.nn as nn
import torch.nn.functional as F

from lcm_explo.domain.models.nn.multihead_attention._exceptions import (
    EmbeddingDimensionDefinitionError,
    EmbeddingDimensionMismatchError,
)


class QKNormedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise EmbeddingDimensionDefinitionError()

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Norm applied per-head (over head_dim) after the Q/K reshape
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, target_length, embed_dim = query.size()
        if embed_dim != self.embed_dim:
            raise EmbeddingDimensionMismatchError()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.contiguous().view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Per-head normalization then scale
        q = self.q_norm(q) * self.scaling
        k = self.k_norm(k)

        attention_output_weights = torch.matmul(q, k.transpose(-2, -1))
        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, self.num_heads, target_length, target_length)
            attention_output_weights += attention_mask

        attention_output_weights = F.softmax(attention_output_weights, dim=-1)
        attention_output_weights = F.dropout(attention_output_weights, p=self.dropout, training=self.training)

        attention_output = torch.matmul(attention_output_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, target_length, embed_dim)

        attention_output = self.out_proj(attention_output)

        return attention_output, attention_output_weights
