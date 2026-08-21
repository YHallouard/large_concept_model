import unittest

import torch

from lcm_explo.domain.models.one_tower_lcm import (
    CosineNoiseSchedule,
    OneTowerLCM,
    OneTowerLCMConfig,
    _build_interleaved_mask,
)


def _small_config(S: int = 4) -> OneTowerLCMConfig:
    return OneTowerLCMConfig(
        concept_embedding_dim=16,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        num_denoising_steps=10,
        num_sampling_steps=3,
        max_seq_len=S * 2,  # must be >= S
        timestep_dim=16,
    )


class TestCosineNoiseSchedule(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = CosineNoiseSchedule(T=10)

    def test_zero_terminal_snr(self) -> None:
        self.assertAlmostEqual(self.schedule.sqrt_alphas[-1].item(), 0.0, places=5)

    def test_monotone_decreasing_alphas(self) -> None:
        a = self.schedule.alphas_cumprod
        self.assertTrue((a[:-1] >= a[1:]).all(), "alphas_cumprod should be monotonically non-increasing")

    def test_q_sample_shape_and_no_nan(self) -> None:
        x0 = torch.randn(2, 4, 16)
        t_idx = torch.randint(1, 10, (2,))
        x_t, eps = self.schedule.q_sample(x0, t_idx)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertFalse(torch.isnan(x_t).any())
        self.assertFalse(torch.isnan(eps).any())

    def test_q_sample_at_t0_close_to_clean(self) -> None:
        x0 = torch.randn(2, 4, 16)
        t_idx = torch.zeros(2, dtype=torch.long)
        x_t, _ = self.schedule.q_sample(x0, t_idx)
        # At t=0, sqrt_alpha[0] ≈ 1, sqrt_one_minus_alpha[0] ≈ 0 → x_t ≈ x0
        self.assertTrue(torch.allclose(x_t, x0, atol=0.01))

    def test_trailing_steps_length(self) -> None:
        steps = self.schedule.trailing_steps(3)
        self.assertEqual(len(steps), 3)


class TestInterleavedMask(unittest.TestCase):
    def setUp(self) -> None:
        self.S = 3
        self.mask = _build_interleaved_mask(self.S, torch.device("cpu")).squeeze()  # (2S, 2S)

    def test_shape(self) -> None:
        raw = _build_interleaved_mask(self.S, torch.device("cpu"))
        self.assertEqual(raw.shape, (1, 1, 2 * self.S, 2 * self.S))

    def test_clean_blocked_from_noisy(self) -> None:
        # Position 2 (clean c_1) must NOT see position 1 (noisy n_1)
        self.assertEqual(self.mask[2, 1].item(), float("-inf"))

    def test_noisy_sees_previous_clean(self) -> None:
        # Position 3 (noisy n_2) must see position 0 (c_0) and position 2 (c_1)
        self.assertEqual(self.mask[3, 0].item(), 0.0)
        self.assertEqual(self.mask[3, 2].item(), 0.0)

    def test_noisy_blocked_from_future(self) -> None:
        # Position 1 (noisy n_1) must NOT see position 2 (c_1, future clean)
        self.assertEqual(self.mask[1, 2].item(), float("-inf"))

    def test_clean_blocked_from_future_clean(self) -> None:
        # Position 0 (c_0) must NOT see position 2 (c_1)
        self.assertEqual(self.mask[0, 2].item(), float("-inf"))

    def test_noisy_sees_self(self) -> None:
        # Odd positions can attend to themselves
        self.assertEqual(self.mask[1, 1].item(), 0.0)
        self.assertEqual(self.mask[3, 3].item(), 0.0)


class TestOneTowerLCM(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _small_config(S=4)
        self.B, self.S, self.D = 2, 4, 16
        self.model = OneTowerLCM(self.config)
        self.x = torch.randn(self.B, self.S, self.D)
        self.y = torch.randn(self.B, self.S, self.D)

    # --- forward (training) ---

    def test_forward_output_keys(self) -> None:
        out = self.model(self.x, self.y)
        self.assertIn("pred_x0", out)
        self.assertIn("x0", out)
        self.assertIn("loss", out)

    def test_forward_pred_x0_shape(self) -> None:
        out = self.model(self.x, self.y)
        self.assertEqual(out["pred_x0"].shape, (self.B, self.S, self.D))

    def test_forward_loss_positive(self) -> None:
        out = self.model(self.x, self.y)
        self.assertGreater(out["loss"].item(), 0.0)

    def test_forward_gradient_flows(self) -> None:
        out = self.model(self.x, self.y)
        out["loss"].backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_drop_attn_changes_output(self) -> None:
        self.model.eval()
        with torch.no_grad():
            t_idx = torch.ones(self.B, dtype=torch.long)
            cond   = self.model(self.x, self.y, t_idx=t_idx, drop_attn=False)["pred_x0"]
            uncond = self.model(self.x, self.y, t_idx=t_idx, drop_attn=True )["pred_x0"]
        self.assertFalse(torch.equal(cond, uncond), "drop_attn=True must change output")

    # --- sampling (inference) ---

    def test_sample_output_shape(self) -> None:
        self.model.eval()
        with torch.no_grad():
            out = self.model.sample(self.x, steps=3)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_sample_no_nan(self) -> None:
        self.model.eval()
        with torch.no_grad():
            out = self.model.sample(self.x, steps=3)
        self.assertFalse(torch.isnan(out).any())

    # --- utilities ---

    def test_normalizer_in_state_dict(self) -> None:
        sd = self.model.state_dict()
        self.assertIn("normalizer.mean", sd)
        self.assertIn("normalizer.std", sd)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self) -> None:
        model = OneTowerLCM(self.config).cuda()
        x = self.x.cuda()
        y = self.y.cuda()
        out = model(x, y)
        self.assertEqual(out["loss"].device.type, "cuda")
