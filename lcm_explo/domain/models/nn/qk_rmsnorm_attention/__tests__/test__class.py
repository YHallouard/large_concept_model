import unittest

import torch

from lcm_explo.domain.models.nn.qk_rmsnorm_attention._class import QKRMSNormSDPAAttention


class TestQKRMSNormSDPAAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_length = 10
        self.num_heads = 4
        self.head_dim = 16

    def test_forward_shape_self_attention(self) -> None:
        # Given a self-attention configuration (q_dim == kv_dim)
        q_dim = 64
        attention = QKRMSNormSDPAAttention(q_dim, q_dim, self.num_heads, self.head_dim)
        x = torch.rand(self.batch_size, self.seq_length, q_dim)

        # When
        output = attention(x, x, x)

        # Then the output lives in the query dimension
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, q_dim))

    def test_forward_shape_cross_attention(self) -> None:
        # Given heterogeneous query/key-value dimensions
        q_dim, kv_dim = 48, 96
        kv_length = 5
        attention = QKRMSNormSDPAAttention(q_dim, kv_dim, self.num_heads, self.head_dim)
        query = torch.rand(self.batch_size, self.seq_length, q_dim)
        key = torch.rand(self.batch_size, kv_length, kv_dim)
        value = torch.rand(self.batch_size, kv_length, kv_dim)

        # When
        output = attention(query, key, value)

        # Then output is projected back to the query dimension
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, q_dim))

    def test_causal_masking_blocks_future(self) -> None:
        # Given a causal self-attention module
        q_dim = 32
        attention = QKRMSNormSDPAAttention(q_dim, q_dim, self.num_heads, self.head_dim)
        attention.eval()
        x = torch.rand(1, self.seq_length, q_dim)

        # When we run causal attention, then perturb a future token
        with torch.no_grad():
            out_full = attention(x, x, x, is_causal=True)
            x_perturbed = x.clone()
            x_perturbed[:, -1, :] += 10.0
            out_perturbed = attention(x_perturbed, x_perturbed, x_perturbed, is_causal=True)

        # Then earlier positions are unchanged (they cannot attend to the future)
        torch.testing.assert_close(out_full[:, :-1], out_perturbed[:, :-1])

    def test_no_nan_output(self) -> None:
        # Given
        q_dim = 64
        attention = QKRMSNormSDPAAttention(q_dim, q_dim, self.num_heads, self.head_dim)
        x = torch.rand(self.batch_size, self.seq_length, q_dim)

        # When
        output = attention(x, x, x, is_causal=True)

        # Then
        self.assertFalse(torch.isnan(output).any())
