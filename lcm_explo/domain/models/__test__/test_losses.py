import unittest

import torch

from lcm_explo.domain.models._losses import BoundaryRatioLoss


class TestBoundaryRatioLoss(unittest.TestCase):
    def test_minimum_at_target_ratio(self) -> None:
        # Given a target ratio R and boundaries/probs both at 1/R
        loss_fn = BoundaryRatioLoss(target_ratio=4.0)
        loss_fn.eval()
        rate = 0.25
        p = torch.full((2, 8), rate)
        b = torch.full((2, 8), rate)

        # When
        loss, f_local, g = loss_fn(p, b)

        # Then the loss is at its minimum (0)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(f_local, torch.tensor(rate))
        torch.testing.assert_close(g, torch.tensor(rate))

    def test_positive_away_from_target(self) -> None:
        # Given
        loss_fn = BoundaryRatioLoss(target_ratio=4.0)
        loss_fn.eval()

        # When boundaries collapse to "everything is a boundary" (no compression)
        loss_all, _, _ = loss_fn(torch.ones(1, 8), torch.ones(1, 8))
        # or to almost none
        loss_none, _, _ = loss_fn(torch.full((1, 8), 0.01), torch.zeros(1, 8))

        # Then the loss is strictly positive at both extremes
        self.assertGreater(loss_all.item(), 0.0)
        self.assertGreater(loss_none.item(), 0.0)

    def test_gradient_only_through_probabilities(self) -> None:
        # Given probs requiring grad and detached boundaries
        loss_fn = BoundaryRatioLoss(target_ratio=4.0)
        loss_fn.eval()
        p = torch.rand(2, 8, requires_grad=True)
        b = torch.bernoulli(torch.full((2, 8), 0.5))

        # When
        loss, _, _ = loss_fn(p, b)
        loss.backward()

        # Then p carries a gradient
        self.assertIsNotNone(p.grad)
        self.assertGreater(p.grad.abs().sum().item(), 0.0)

    def test_ema_updates_in_train_and_frozen_in_eval(self) -> None:
        # Given
        loss_fn = BoundaryRatioLoss(target_ratio=4.0, warmup_updates=0)

        # When training with a high boundary rate
        loss_fn.train()
        before = loss_fn.f_ema.clone()
        loss_fn(torch.full((1, 8), 0.9), torch.ones(1, 8))
        after_train = loss_fn.f_ema.clone()

        # and then in eval
        loss_fn.eval()
        loss_fn(torch.full((1, 8), 0.9), torch.ones(1, 8))
        after_eval = loss_fn.f_ema.clone()

        # Then the EMA moves during training and is frozen in eval
        self.assertGreater((after_train - before).abs().item(), 0.0)
        torch.testing.assert_close(after_eval, after_train)

    def test_ema_buffer_in_state_dict(self) -> None:
        # Given
        loss_fn = BoundaryRatioLoss(target_ratio=4.0)

        # When
        state = loss_fn.state_dict()

        # Then the EMA survives checkpointing
        self.assertIn("f_ema", state)
        self.assertIn("num_updates", state)
