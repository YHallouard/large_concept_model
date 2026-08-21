import unittest
from unittest.mock import MagicMock, patch

import torch

from lcm_explo.domain.models.dlcm import DLCM, DLCMConfig
from lcm_explo.domain.usecases.warm_start import _project_embedding_pca, warm_start_embedding_from_gpt2


class TestProjectEmbeddingPCA(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_output_shape(self) -> None:
        # Given a source embedding table larger than the target dimension
        source = torch.randn(200, 32)

        # When
        projected = _project_embedding_pca(source, d_token=8, target_std=0.02)

        # Then it is reduced to (vocab, d_token)
        self.assertEqual(projected.shape, (200, 8))

    def test_rescaled_to_target_std(self) -> None:
        # Given a source table with an arbitrary, large scale
        source = torch.randn(200, 32) * 5.0

        # When
        projected = _project_embedding_pca(source, d_token=8, target_std=0.02)

        # Then the projected table's std matches the target (keeps the residual
        # stream at the same order of magnitude as the model's random init).
        torch.testing.assert_close(projected.std(), torch.tensor(0.02), atol=1e-4, rtol=0)

    def test_raises_when_d_token_exceeds_source_dim(self) -> None:
        # Given a target dimension larger than the source
        source = torch.randn(50, 16)

        # When / Then
        with self.assertRaises(ValueError):
            _project_embedding_pca(source, d_token=32, target_std=0.02)

    def test_deterministic_given_seed(self) -> None:
        # Given the same source and a fixed seed
        source = torch.randn(200, 32)

        # When run twice under the same seed
        torch.manual_seed(1)
        first = _project_embedding_pca(source, d_token=8, target_std=0.02)
        torch.manual_seed(1)
        second = _project_embedding_pca(source, d_token=8, target_std=0.02)

        # Then results match
        torch.testing.assert_close(first, second)


def _fake_gpt2(vocab_size: int, hidden_size: int) -> MagicMock:
    fake = MagicMock()
    fake.wte.weight.data = torch.randn(vocab_size, hidden_size)
    return fake


def _tiny_model(vocab_size: int, d_token: int = 16, tie_embeddings: bool = True) -> DLCM:
    config = DLCMConfig(
        vocab_size=vocab_size,
        d_token=d_token,
        d_concept=32,
        d_scan=8,
        num_encoder_layers=1,
        num_backbone_layers=1,
        num_decoder_layers=1,
        num_token_heads=2,
        num_concept_heads=2,
        ffn_multiplier=2,
        max_seq_len=16,
        tie_embeddings=tie_embeddings,
    )
    return DLCM(config)


class TestWarmStartEmbeddingFromGPT2(unittest.TestCase):
    @patch("transformers.GPT2Model.from_pretrained")
    def test_applies_and_updates_tied_embedding(self, mock_from_pretrained: MagicMock) -> None:
        # Given a model whose vocab matches the (mocked) GPT-2 source
        vocab_size = 64
        model = _tiny_model(vocab_size, d_token=16)
        mock_from_pretrained.return_value = _fake_gpt2(vocab_size, hidden_size=32)
        before = model.encoder.embedding.weight.clone()

        # When
        applied = warm_start_embedding_from_gpt2(model)

        # Then the (tied) embedding changed and the LM head shares the update
        self.assertTrue(applied)
        self.assertFalse(torch.allclose(model.encoder.embedding.weight, before))
        self.assertIs(model.lm_head.weight, model.encoder.embedding.weight)

    @patch("transformers.GPT2Model.from_pretrained")
    def test_updates_untied_lm_head_too(self, mock_from_pretrained: MagicMock) -> None:
        # Given an untied model
        vocab_size = 64
        model = _tiny_model(vocab_size, d_token=16, tie_embeddings=False)
        mock_from_pretrained.return_value = _fake_gpt2(vocab_size, hidden_size=32)

        # When
        warm_start_embedding_from_gpt2(model)

        # Then both the embedding and the LM head were warm-started identically
        torch.testing.assert_close(model.encoder.embedding.weight, model.lm_head.weight)

    @patch("transformers.GPT2Model.from_pretrained")
    def test_skips_on_vocab_mismatch(self, mock_from_pretrained: MagicMock) -> None:
        # Given a model whose vocab does NOT match the source
        model = _tiny_model(vocab_size=64, d_token=16)
        mock_from_pretrained.return_value = _fake_gpt2(vocab_size=100, hidden_size=32)
        before = model.encoder.embedding.weight.clone()

        # When
        applied = warm_start_embedding_from_gpt2(model)

        # Then nothing changed
        self.assertFalse(applied)
        torch.testing.assert_close(model.encoder.embedding.weight, before)
