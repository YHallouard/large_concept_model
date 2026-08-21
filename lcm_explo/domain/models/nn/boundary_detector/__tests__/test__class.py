import unittest

import torch

from lcm_explo.domain.models.nn.boundary_detector._class import BoundaryDetector


class TestBoundaryDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.seq_length = 12
        self.d_token = 16
        self.d_scan = 8

    def test_forward_shape_and_range(self) -> None:
        # Given
        detector = BoundaryDetector(self.d_token, self.d_scan)
        h = torch.rand(self.batch_size, self.seq_length, self.d_token)

        # When
        p = detector(h)

        # Then p has shape (B, L) and lives in [0, 1]
        self.assertEqual(p.shape, (self.batch_size, self.seq_length))
        self.assertTrue((p >= 0.0).all())
        self.assertTrue((p <= 1.0).all())

    def test_first_token_is_always_boundary(self) -> None:
        # Given
        detector = BoundaryDetector(self.d_token, self.d_scan)
        h = torch.rand(self.batch_size, self.seq_length, self.d_token)

        # When
        p = detector(h)

        # Then p_1 == 1 exactly
        torch.testing.assert_close(p[:, 0], torch.ones(self.batch_size))

    def test_sample_is_binary_and_gradient_free(self) -> None:
        # Given
        detector = BoundaryDetector(self.d_token, self.d_scan)
        h = torch.rand(self.batch_size, self.seq_length, self.d_token, requires_grad=True)
        p = detector(h)

        # When
        b = detector.sample(p, temperature=0.5)

        # Then boundaries are in {0, 1}, first column is 1, and carry no gradient
        self.assertTrue(((b == 0.0) | (b == 1.0)).all())
        torch.testing.assert_close(b[:, 0], torch.ones(self.batch_size))
        self.assertIsNone(b.grad_fn)

    def test_threshold_is_deterministic(self) -> None:
        # Given
        detector = BoundaryDetector(self.d_token, self.d_scan)
        h = torch.rand(self.batch_size, self.seq_length, self.d_token)
        p = detector(h)

        # When
        b1 = detector.threshold(p)
        b2 = detector.threshold(p)

        # Then
        torch.testing.assert_close(b1, b2)
        self.assertTrue(((b1 == 0.0) | (b1 == 1.0)).all())

    def test_rule_mode_has_no_parameters(self) -> None:
        # Given a rule-based detector
        detector = BoundaryDetector(self.d_token, self.d_scan, mode="rule")

        # When we count its parameters
        num_params = sum(p.numel() for p in detector.parameters())

        # Then there are none (segmentation decoupled from learning)
        self.assertEqual(num_params, 0)

    def test_gradient_reaches_projections_via_p(self) -> None:
        # Given a learned detector
        detector = BoundaryDetector(self.d_token, self.d_scan)
        h = torch.rand(self.batch_size, self.seq_length, self.d_token)

        # When we backprop through the continuous probabilities
        p = detector(h)
        p.sum().backward()

        # Then W_q receives a gradient (the only path allowed to it)
        self.assertIsNotNone(detector.w_q.weight.grad)
        self.assertGreater(detector.w_q.weight.grad.abs().sum().item(), 0.0)
