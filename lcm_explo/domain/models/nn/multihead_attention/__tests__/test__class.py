import unittest

import torch

from lcm_explo.domain.models.nn.multihead_attention._class import QKNormedMultiheadAttention
from lcm_explo.domain.models.nn.multihead_attention._exceptions import (
    EmbeddingDimensionDefinitionError,
    EmbeddingDimensionMismatchError,
)


class TestQKNormedMultiheadAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 64
        self.num_heads = 8
        self.batch_size = 2
        self.seq_length = 10
        self.attention = QKNormedMultiheadAttention(self.embed_dim, self.num_heads)

    def test_forward_shape(self) -> None:
        # Given
        query = torch.rand(self.batch_size, self.seq_length, self.embed_dim)
        key = torch.rand(self.batch_size, self.seq_length, self.embed_dim)
        value = torch.rand(self.batch_size, self.seq_length, self.embed_dim)

        # When
        output, weights = self.attention(query, key, value)

        # Then
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))

    def test_embedding_dimension_definition_error(self) -> None:
        # Given
        # When & Then
        with self.assertRaises(EmbeddingDimensionDefinitionError):
            QKNormedMultiheadAttention(embed_dim=65, num_heads=self.num_heads)

    def test_embedding_dimension_mismatch_error(self) -> None:
        # Given
        query = torch.rand(self.batch_size, self.seq_length, self.embed_dim + 1)
        key = torch.rand(self.batch_size, self.seq_length, self.embed_dim)
        value = torch.rand(self.batch_size, self.seq_length, self.embed_dim)

        # When & Then
        with self.assertRaises(EmbeddingDimensionMismatchError):
            self.attention(query, key, value)

    def test_forward_with_attention_mask(self) -> None:
        # Given
        query = torch.rand(self.batch_size, self.seq_length, self.embed_dim)
        key = torch.rand(self.batch_size, self.seq_length, self.embed_dim)
        value = torch.rand(self.batch_size, self.seq_length, self.embed_dim)

        padding_mask = torch.ones(self.batch_size, self.seq_length)
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attention_mask[:, :, :, : self.seq_length // 2] = 0
        attention_mask = (1.0 - attention_mask) * -10000.0

        # When
        output, weights = self.attention(query, key, value, attention_mask=attention_mask)

        # Then
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))
