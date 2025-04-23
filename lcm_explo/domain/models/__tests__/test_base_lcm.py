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
    StandardScaler,
    UnknowNormTypeError,
)
from lcm_explo.domain.models.nn import DyT


@unittest.skip("Skipping pytest fixtures for unittest")
def config() -> BaseLCMConfig:
    return BaseLCMConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        concept_embedding_dim=32,
    )


@unittest.skip("Skipping pytest fixtures for unittest")
def batch_size() -> int:
    return 4


@unittest.skip("Skipping pytest fixtures for unittest")
def sequence_length() -> int:
    return 16


@unittest.skip("Skipping pytest fixtures for unittest")
def concept_embeddings(batch_size: int, sequence_length: int, config: BaseLCMConfig) -> torch.Tensor:
    return torch.randn(batch_size, sequence_length, config.concept_embedding_dim)


class TestStandardScaler(unittest.TestCase):
    def test_forward(self) -> None:
        scaler = StandardScaler()
        x = torch.randn(4, 16, 32)
        output = scaler(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_running_stats(self) -> None:
        scaler = StandardScaler()
        x = torch.randn(4, 16, 32)

        # Initial stats
        initial_mean = scaler.running_mean.clone()
        initial_var = scaler.running_var.clone()

        # Forward pass in training mode
        scaler.train()
        _ = scaler(x)

        # Check that running stats were updated
        self.assertFalse(torch.equal(scaler.running_mean, initial_mean))
        self.assertFalse(torch.equal(scaler.running_var, initial_var))

    def test_eval_mode(self) -> None:
        scaler = StandardScaler()
        x = torch.randn(4, 16, 32)

        # Set to eval mode
        scaler.eval()
        initial_mean = scaler.running_mean.clone()
        initial_var = scaler.running_var.clone()

        # Forward pass
        _ = scaler(x)

        # Check that running stats were not updated in eval mode
        self.assertTrue(torch.equal(scaler.running_mean, initial_mean))
        self.assertTrue(torch.equal(scaler.running_var, initial_var))


class TestBaseLCMPreNet(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        concept_embeddings = torch.randn(4, 16, 32)
        pre_net = BaseLCMPreNet(config)
        output, scaler = pre_net(concept_embeddings)

        # Check output shape
        self.assertEqual(
            output.shape,
            (
                concept_embeddings.shape[0],
                concept_embeddings.shape[1],
                config.hidden_size,
            ),
        )

        # Check scaler is returned
        self.assertIsInstance(scaler, StandardScaler)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestBaseLCMPostNet(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        concept_embeddings = torch.randn(4, 16, 32)
        pre_net = BaseLCMPreNet(config)
        post_net = BaseLCMPostNet(config)

        # Get normalized input and scaler from pre_net
        hidden, scaler = pre_net(concept_embeddings)

        # Process through post_net
        output = post_net(hidden, scaler)

        # Check output shape matches input shape
        self.assertEqual(output.shape, concept_embeddings.shape)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestBaseLCMDecoderLayer(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        concept_embeddings = torch.randn(4, 16, 32)
        layer = BaseLCMDecoderLayer(config)

        # Get normalized input from pre_net
        pre_net = BaseLCMPreNet(config)
        hidden, _ = pre_net(concept_embeddings)

        # Create a padding mask
        padding_mask = torch.ones(hidden.shape[:2], dtype=torch.float32)

        output = layer(hidden, padding_mask)

        # Check output shape
        self.assertEqual(output.shape, hidden.shape)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_layer_norm_switch(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        # Test with RMSNorm
        config.norm_type = "RMSNorm"
        layer = BaseLCMDecoderLayer(config)
        self.assertIsInstance(layer.self_attention_layer_norm, nn.LayerNorm)
        self.assertIsInstance(layer.feed_forward_layer_norm, nn.LayerNorm)

        # Test with DyT
        config.norm_type = "DyT"
        layer = BaseLCMDecoderLayer(config)
        self.assertIsInstance(layer.self_attention_layer_norm, DyT)
        self.assertIsInstance(layer.feed_forward_layer_norm, DyT)

        # Test with unknown norm type
        config.norm_type = "UnknownNorm"  # type: ignore  # noqa: PGH003
        with self.assertRaises(UnknowNormTypeError) as context:
            layer = BaseLCMDecoderLayer(config)
        self.assertEqual(str(context.exception), "Unknown norm type: UnknownNorm")


class TestBaseLCMDecoder(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        concept_embeddings = torch.randn(4, 16, 32)
        decoder = BaseLCMDecoder(config)

        # Get normalized input from pre_net
        pre_net = BaseLCMPreNet(config)
        hidden, _ = pre_net(concept_embeddings)

        # Create a padding mask
        padding_mask = torch.ones(hidden.shape[:2], dtype=torch.float32)

        output = decoder(hidden, padding_mask)

        # Check output shape
        self.assertEqual(output.shape, hidden.shape)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestBaseLCM(unittest.TestCase):
    def test_forward(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        concept_embeddings = torch.randn(4, 16, 32)
        model = BaseLCM(config)

        # Create a padding mask
        padding_mask = torch.ones(concept_embeddings.shape[:2], dtype=torch.float32)

        output = model(concept_embeddings, padding_mask)

        # Check output shape matches input shape
        self.assertEqual(output.shape, concept_embeddings.shape)

        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_weight_initialization(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        model = BaseLCM(config)

        # Check linear layer initialization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Check weight initialization
                mean = module.weight.mean().item()
                std = module.weight.std().item()
                self.assertAlmostEqual(mean, 0, delta=0.1)  # Mean should be close to 0
                self.assertAlmostEqual(
                    std, config.initializer_range, delta=0.1
                )  # Std should be close to initializer_range

                # Check bias initialization if it exists
                if module.bias is not None:
                    self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))

            elif isinstance(module, nn.LayerNorm):
                # Check LayerNorm initialization
                self.assertTrue(torch.allclose(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))

    def test_next_concept_prediction(self) -> None:
        config = BaseLCMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            concept_embedding_dim=32,
        )
        # Create a sequence of concept embeddings
        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, config.concept_embedding_dim)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        model = BaseLCM(config)
        output = model(x, padding_mask)

        # Check that output has same shape as input
        self.assertEqual(output.shape, x.shape)

        # Check that output is different from input (model transforms the sequence)
        self.assertFalse(torch.allclose(output, x))
