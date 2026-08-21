import unittest

import torch

from lcm_explo.domain.models._losses import BoundaryRatioLoss
from lcm_explo.domain.models.dlcm import DLCM, DLCMConfig


def _tiny_config(**overrides: object) -> DLCMConfig:
    base = dict(
        vocab_size=97,
        d_token=32,
        d_concept=64,
        d_scan=16,
        num_encoder_layers=2,
        num_backbone_layers=2,
        num_decoder_layers=1,
        num_token_heads=4,
        num_concept_heads=4,
        ffn_multiplier=2,
        max_seq_len=32,
    )
    base.update(overrides)
    return DLCMConfig(**base)  # type: ignore[arg-type]


class TestDLCM(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_length = 16
        torch.manual_seed(0)

    def _input_ids(self, config: DLCMConfig) -> torch.Tensor:
        return torch.randint(0, config.vocab_size, (self.batch_size, self.seq_length))

    def test_forward_smoke(self) -> None:
        # Given a tiny model in train mode
        config = _tiny_config()
        model = DLCM(config)
        input_ids = self._input_ids(config)

        # When
        out = model(input_ids)

        # Then logits have the expected shape and are finite
        self.assertEqual(out.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isinf(out.logits).any())
        self.assertEqual(out.boundary_probs.shape, (self.batch_size, self.seq_length))

    def test_eval_forward_is_deterministic(self) -> None:
        # Given a model in eval mode (hard-threshold boundaries)
        config = _tiny_config()
        model = DLCM(config)
        model.eval()
        input_ids = self._input_ids(config)

        # When run twice
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)

        # Then outputs match (no Bernoulli sampling)
        torch.testing.assert_close(out1.logits, out2.logits)

    def test_embeddings_are_tied(self) -> None:
        # Given tie_embeddings=True
        config = _tiny_config(tie_embeddings=True)
        model = DLCM(config)

        # Then the LM head shares the embedding matrix
        self.assertIs(model.lm_head.weight, model.encoder.embedding.weight)

    def test_gradient_routing_ce_does_not_reach_boundary_projections(self) -> None:
        # Given a model and CE-only loss
        config = _tiny_config()
        model = DLCM(config)
        model.train()
        input_ids = self._input_ids(config)
        labels = self._input_ids(config)

        # When we backprop cross-entropy alone
        out = model(input_ids)
        ce = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, config.vocab_size), labels.reshape(-1)
        )
        ce.backward()

        # Then the boundary projections receive NO gradient (decoupled segmentation),
        # while the encoder embedding does (pooling passes CE back to it).
        w_q_grad = model.segmenter.boundary_detector.w_q.weight.grad
        self.assertTrue(w_q_grad is None or w_q_grad.abs().sum().item() == 0.0)
        self.assertIsNotNone(model.encoder.embedding.weight.grad)
        self.assertGreater(model.encoder.embedding.weight.grad.abs().sum().item(), 0.0)

    def test_gradient_routing_aux_reaches_boundary_projections(self) -> None:
        # Given a model and the auxiliary boundary-ratio loss
        config = _tiny_config()
        model = DLCM(config)
        model.train()
        aux_loss = BoundaryRatioLoss(target_ratio=config.target_ratio)
        input_ids = self._input_ids(config)

        # When we backprop the aux loss
        out = model(input_ids)
        aux, _, _ = aux_loss(out.boundary_probs, out.boundaries)
        aux.backward()

        # Then the boundary projections receive a gradient (via G = mean(p))
        w_q_grad = model.segmenter.boundary_detector.w_q.weight.grad
        self.assertIsNotNone(w_q_grad)
        self.assertGreater(w_q_grad.abs().sum().item(), 0.0)

    def test_generate_extends_sequence(self) -> None:
        # Given a model and a short prompt
        config = _tiny_config()
        model = DLCM(config)
        prompt = self._input_ids(config)

        # When
        out = model.generate(prompt, max_new_tokens=5, top_k=10)

        # Then the sequence grew by max_new_tokens
        self.assertEqual(out.shape, (self.batch_size, self.seq_length + 5))
