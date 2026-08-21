import unittest

import torch

from lcm_explo.domain.models.nn.concept_smoothing._class import CausalConceptSmoothing


class TestCausalConceptSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.num_concepts = 8
        self.d_concept = 16

    def test_shape_preserved(self) -> None:
        # Given
        smoothing = CausalConceptSmoothing(self.d_concept, kernel_size=3)
        z = torch.rand(self.batch_size, self.num_concepts, self.d_concept)

        # When
        out = smoothing(z)

        # Then
        self.assertEqual(out.shape, z.shape)

    def test_causality_over_concept_axis(self) -> None:
        # Given a smoothing module in eval mode
        smoothing = CausalConceptSmoothing(self.d_concept, kernel_size=3)
        smoothing.eval()
        z = torch.rand(1, self.num_concepts, self.d_concept)
        k = 4

        # When we perturb concept k
        with torch.no_grad():
            out = smoothing(z)
            z_perturbed = z.clone()
            z_perturbed[:, k, :] += 5.0
            out_perturbed = smoothing(z_perturbed)

        # Then earlier concepts (< k) are unchanged; concept k itself changes
        torch.testing.assert_close(out[:, :k], out_perturbed[:, :k])
        self.assertGreater((out[:, k] - out_perturbed[:, k]).abs().sum().item(), 0.0)
