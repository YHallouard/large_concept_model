import unittest

import torch
from torch import nn

from lcm_explo.domain.models.base_lcm import (
    BaseLCM,
    BaseLCMConfig,
    BaseLCMDecoder,
    BaseLCMDecoderLayer,
    BaseLCMPostNet,
    BaseLCMPreNet,
    Normalizer,
    UnknowNormTypeError,
)
from lcm_explo.domain.models.nn import DyT


class TestNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 32
        self.normalizer = Normalizer(self.dim)

    def test_normalize_per_dimension(self) -> None:
        mean = torch.arange(self.dim, dtype=torch.float32)
        std = torch.ones(self.dim) * 2.0
        self.normalizer.load_stats(mean, std)

        x = torch.zeros(1, 1, self.dim)
        normed = self.normalizer.normalize(x)
        expected = -mean / (std + 1e-8)
        self.assertTrue(torch.allclose(normed.squeeze(), expected, atol=1e-5))

    def test_denormalize_roundtrip(self) -> None:
        mean = torch.randn(self.dim)
        std = torch.abs(torch.randn(self.dim)) + 0.5
        self.normalizer.load_stats(mean, std)

        x = torch.randn(4, 8, self.dim)
        recovered = self.normalizer.denormalize(self.normalizer.normalize(x))
        self.assertTrue(torch.allclose(recovered, x, atol=1e-5))


class TestBaseLCMPreNet(unittest.TestCase):
    def test_forward_shape(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64, max_seq_len=16, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32,
        )
        x = torch.randn(4, 16, 32)
        pre_net = BaseLCMPreNet(config)
        output = pre_net(x)
        self.assertEqual(output.shape, (4, 16, 64))
        self.assertFalse(torch.isnan(output).any())


class TestBaseLCMPostNet(unittest.TestCase):
    def test_forward_shape(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64, max_seq_len=16, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32,
        )
        hidden = torch.randn(4, 16, 64)
        post_net = BaseLCMPostNet(config)
        output = post_net(hidden)
        self.assertEqual(output.shape, (4, 16, 32))
        self.assertFalse(torch.isnan(output).any())


class TestBaseLCMDecoderLayer(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64, max_seq_len=16, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32,
        )
        hidden = torch.randn(4, 16, 64)
        padding_mask = torch.ones(4, 16)
        layer = BaseLCMDecoderLayer(config)
        output = layer(hidden, padding_mask)
        self.assertEqual(output.shape, hidden.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_layer_norm_switch(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64, max_seq_len=16, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32, norm_type="RMSNorm",
        )
        layer = BaseLCMDecoderLayer(config)
        self.assertIsInstance(layer.self_attention_layer_norm, nn.LayerNorm)

        config.norm_type = "DyT"
        layer = BaseLCMDecoderLayer(config)
        self.assertIsInstance(layer.self_attention_layer_norm, DyT)

        config.norm_type = "UnknownNorm"  # type: ignore
        with self.assertRaises(UnknowNormTypeError):
            BaseLCMDecoderLayer(config)


class TestBaseLCMDecoder(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64, max_seq_len=16, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32,
        )
        hidden = torch.randn(4, 16, 64)
        padding_mask = torch.ones(4, 16)
        decoder = BaseLCMDecoder(config)
        output = decoder(hidden, padding_mask)
        self.assertEqual(output.shape, hidden.shape)
        self.assertFalse(torch.isnan(output).any())


class TestBaseLCM(unittest.TestCase):
    def _make_config(self, seq_len: int = 16) -> BaseLCMConfig:
        return BaseLCMConfig(
            hidden_size=64, max_seq_len=seq_len, num_attention_heads=4, num_hidden_layers=2,
            intermediate_size=128, concept_embedding_dim=32,
        )

    def test_forward_returns_sonar_space(self) -> None:
        """forward() must return raw SONAR space (denormalized) — paper §2.3.1 eq. 2."""
        config = self._make_config(4)
        x = torch.randn(2, 4, 32)
        padding_mask = torch.ones(2, 4)
        model = BaseLCM(config)

        # Load non-trivial stats so we can detect whether denorm was applied
        median = torch.ones(32) * 5.0
        iqr = torch.ones(32) * 2.0
        model.normalizer.load_stats(median, iqr)

        y = model.forward(x, padding_mask)
        self.assertEqual(y.shape, x.shape)
        self.assertFalse(torch.isnan(y).any())

        # The output must NOT be in normalized space.  In normalized space every
        # dimension would be ~O(1); after denormalize it is shifted by median ≈ 5.
        # Check that the mean absolute value is clearly above 1.
        self.assertGreater(y.abs().mean().item(), 1.0)

    def test_predict_is_alias_for_forward(self) -> None:
        """predict() and forward() must return identical results (paper: both SONAR space)."""
        config = self._make_config(4)
        x = torch.randn(2, 4, 32)
        padding_mask = torch.ones(2, 4)
        model = BaseLCM(config)
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model.forward(x, padding_mask), model.predict(x, padding_mask)))

    def test_normalizer_stats_in_state_dict(self) -> None:
        config = self._make_config()
        model = BaseLCM(config)
        sd = model.state_dict()
        self.assertIn("normalizer.mean", sd)
        self.assertIn("normalizer.std", sd)

    def test_weight_initialization(self) -> None:
        config = self._make_config()
        model = BaseLCM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                mean = module.weight.mean().item()
                self.assertAlmostEqual(mean, 0, delta=0.2)
                if module.bias is not None:
                    self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))
