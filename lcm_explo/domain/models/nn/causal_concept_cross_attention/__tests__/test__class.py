import unittest

import torch

from lcm_explo.domain.models.nn.causal_concept_cross_attention._class import CausalConceptCrossAttention
from lcm_explo.domain.models.nn.segment_pooling._class import segment_ids_from_boundaries


class TestCausalConceptCrossAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.d_token = 24
        self.d_concept = 32
        self.num_heads = 4
        self.head_dim = 8
        # segments: {0,1,2}, {3,4}, {5}, {6,7} -> 4 concepts
        self.b = torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float32)
        self.seg_id, self.num_segments = segment_ids_from_boundaries(self.b)
        self.seq_length = self.b.size(1)
        self.max_concepts = int(self.num_segments.max().item())

    def _module(self) -> CausalConceptCrossAttention:
        module = CausalConceptCrossAttention(self.d_token, self.d_concept, self.num_heads, self.head_dim)
        module.eval()
        return module

    def test_output_shape_and_finite(self) -> None:
        # Given
        module = self._module()
        h = torch.rand(1, self.seq_length, self.d_token)
        concepts = torch.rand(1, self.max_concepts, self.d_concept)

        # When
        out = module(h, concepts, self.seg_id)

        # Then output is in the token stream, finite (BOS path avoids NaN rows)
        self.assertEqual(out.shape, (1, self.seq_length, self.d_token))
        self.assertFalse(torch.isnan(out).any())

    def test_no_leakage_from_future_concepts(self) -> None:
        # Given a token attends only to concepts of completed segments
        module = self._module()
        h = torch.rand(1, self.seq_length, self.d_token)
        concepts = torch.rand(1, self.max_concepts, self.d_concept)
        k = 1  # perturb concept c_1

        # When we perturb concept c_k
        with torch.no_grad():
            out = module(h, concepts, self.seg_id)
            concepts_perturbed = concepts.clone()
            concepts_perturbed[:, k, :] += 7.0
            out_perturbed = module(h, concepts_perturbed, self.seg_id)

        # Then tokens whose segment id <= k are unaffected (they cannot see c_k),
        # while at least one token with seg_id > k changes.
        unaffected = self.seg_id[0] <= k
        affected = self.seg_id[0] > k
        torch.testing.assert_close(out[0][unaffected], out_perturbed[0][unaffected])
        self.assertGreater((out[0][affected] - out_perturbed[0][affected]).abs().sum().item(), 0.0)

    def test_is_causal_matches_explicit_mask(self) -> None:
        # Given the same replicated key/value bank
        module = self._module()
        h = torch.rand(1, self.seq_length, self.d_token)
        concepts = torch.rand(1, self.max_concepts, self.d_concept)

        # When we compare is_causal=True against an explicit lower-triangular mask
        with torch.no_grad():
            out = module(h, concepts, self.seg_id)
            z_rep = module._replicate(concepts, self.seg_id)
            causal_mask = torch.tril(torch.ones(self.seq_length, self.seq_length, dtype=torch.bool))
            ref_attended = module.attention(h, z_rep, z_rep, attn_mask=causal_mask, is_causal=False)
            ref = ref_attended + h

        # Then they are numerically identical
        torch.testing.assert_close(out, ref)

    def test_gradient_flows_to_concepts(self) -> None:
        # Given
        module = self._module()
        h = torch.rand(1, self.seq_length, self.d_token)
        concepts = torch.rand(1, self.max_concepts, self.d_concept, requires_grad=True)

        # When we backprop through the attended output
        out = module(h, concepts, self.seg_id)
        out.sum().backward()

        # Then the concept backbone receives a gradient
        self.assertIsNotNone(concepts.grad)
        self.assertGreater(concepts.grad.abs().sum().item(), 0.0)
