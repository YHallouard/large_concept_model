import unittest
from unittest.mock import MagicMock, patch

from lcm_explo.workflows._flows import _needs_offload, _build_strategy

_LARGE = 618_000_000  # ~large One-Tower preset, ~9.9 GB fp32
_BASE  = 155_000_000  # ~base preset, ~2.48 GB fp32


class TestNeedsOffload(unittest.TestCase):
    def test_no_offload_when_budget_none(self) -> None:
        self.assertFalse(_needs_offload(_LARGE, None, "gpu"))

    def test_no_offload_on_cpu(self) -> None:
        self.assertFalse(_needs_offload(_LARGE, 1.0, "cpu"))

    def test_no_offload_on_mps(self) -> None:
        self.assertFalse(_needs_offload(_LARGE, 1.0, "mps"))

    def test_no_offload_within_budget(self) -> None:
        # base preset ~2.48 GB < 11.0 GB
        self.assertFalse(_needs_offload(_BASE, 11.0, "gpu"))

    def test_offload_over_budget(self) -> None:
        # large preset ~9.9 GB > 8.0 GB
        self.assertTrue(_needs_offload(_LARGE, 8.0, "gpu"))

    def test_boundary_exactly_at_budget(self) -> None:
        # exactly at budget → no offload (strict >)
        n = int(8.0 * 1e9 / 16)
        self.assertFalse(_needs_offload(n, 8.0, "gpu"))

    def test_one_byte_over_budget(self) -> None:
        n = int(8.0 * 1e9 / 16) + 1
        self.assertTrue(_needs_offload(n, 8.0, "gpu"))


class TestBuildStrategy(unittest.TestCase):
    def test_returns_auto_below_budget(self) -> None:
        result = _build_strategy(_BASE, 11.0, "gpu")
        self.assertEqual(result, "auto")

    def test_returns_auto_on_cpu(self) -> None:
        result = _build_strategy(_LARGE, 1.0, "cpu")
        self.assertEqual(result, "auto")

    def test_returns_auto_when_no_budget(self) -> None:
        result = _build_strategy(_LARGE, None, "gpu")
        self.assertEqual(result, "auto")

    def test_raises_import_error_without_deepspeed(self) -> None:
        # Simulate DeepSpeed not installed by making the import fail
        with patch.dict("sys.modules", {"lightning.pytorch.strategies": None}):
            with self.assertRaises((ImportError, TypeError)):
                _build_strategy(_LARGE, 8.0, "gpu")

    def test_returns_deepspeed_strategy_when_available(self) -> None:
        mock_strategy = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy)

        with patch("lcm_explo.workflows._flows._needs_offload", return_value=True):
            with patch.dict("sys.modules", {}):
                import sys
                fake_module = MagicMock()
                fake_module.DeepSpeedStrategy = mock_strategy_class
                original = sys.modules.get("lightning.pytorch.strategies")
                sys.modules["lightning.pytorch.strategies"] = fake_module
                try:
                    result = _build_strategy(_LARGE, 8.0, "gpu")
                    mock_strategy_class.assert_called_once_with(
                        stage=2,
                        offload_optimizer=True,
                        allgather_bucket_size=5e8,
                        reduce_bucket_size=5e8,
                    )
                    self.assertEqual(result, mock_strategy)
                finally:
                    if original is None:
                        del sys.modules["lightning.pytorch.strategies"]
                    else:
                        sys.modules["lightning.pytorch.strategies"] = original
