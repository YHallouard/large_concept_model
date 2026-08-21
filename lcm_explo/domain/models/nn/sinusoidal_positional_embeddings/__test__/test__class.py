import unittest

import torch

from lcm_explo.domain.models.nn.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding


class TestSinusoidalPositionalEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 64
        self.max_seq_len = 128
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, self.max_seq_len)

    def test_forward_shape(self) -> None:
        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, self.d_model)
        output = self.embedding(x)
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_forward_values(self) -> None:
        batch_size = 2
        seq_len = 128
        x = torch.zeros(batch_size, seq_len, self.d_model)
        output = self.embedding(x)
        self.assertFalse(torch.all(output == 0))
