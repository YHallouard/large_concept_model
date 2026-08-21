import unittest

import torch

from lcm_explo.domain.models._presets import base_lcm_config_for_size
from lcm_explo.domain.models.base_lcm import BaseLCM, BaseLCMConfig
from lcm_explo.domain.models.dlcm import DLCMConfig
from lcm_explo.utils._model_size import count_parameters, format_parameter_count
from lcm_explo.workflows._inputs import (
    CustomDLCMModelSpec,
    CustomModelSpec,
    PresetDLCMModelSpec,
    PresetModelSpec,
    resolve_base_lcm_config,
    resolve_dlcm_config,
)


class TestBaseConfigForSize(unittest.TestCase):
    def test_tiny_preset(self) -> None:
        config = base_lcm_config_for_size("tiny")
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 4)

    def test_base_preset(self) -> None:
        config = base_lcm_config_for_size("base")
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_hidden_layers, 12)

    def test_custom_max_seq_len(self) -> None:
        config = base_lcm_config_for_size("small", max_seq_len=64)
        self.assertEqual(config.max_seq_len, 64)

    def test_head_dim_divisible(self) -> None:
        for size in ("tiny", "small", "base", "large"):
            config = base_lcm_config_for_size(size)  # type: ignore[arg-type]
            self.assertEqual(config.hidden_size % config.num_attention_heads, 0, size)


class TestResolveBaseLCMConfig(unittest.TestCase):
    def test_preset_spec_returns_config(self) -> None:
        spec = PresetModelSpec(size="tiny")
        config = resolve_base_lcm_config(spec)
        self.assertIsInstance(config, BaseLCMConfig)
        self.assertEqual(config.hidden_size, 256)

    def test_custom_spec_returns_config(self) -> None:
        custom = BaseLCMConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
        spec = CustomModelSpec(config=custom)
        config = resolve_base_lcm_config(spec)
        self.assertIs(config, custom)

    def test_unknown_spec_raises(self) -> None:
        with self.assertRaises(TypeError):
            resolve_base_lcm_config(object())


class TestCountParameters(unittest.TestCase):
    def test_count_is_positive(self) -> None:
        config = base_lcm_config_for_size("tiny")
        config.concept_embedding_dim = 32
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 1
        config.num_attention_heads = 4
        model = BaseLCM(config)
        n = count_parameters(model)
        self.assertGreater(n, 0)

    def test_all_params_vs_trainable(self) -> None:
        config = base_lcm_config_for_size("tiny")
        config.concept_embedding_dim = 32
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 1
        config.num_attention_heads = 4
        model = BaseLCM(config)
        n_trainable = count_parameters(model, trainable_only=True)
        n_all = count_parameters(model, trainable_only=False)
        self.assertGreaterEqual(n_all, n_trainable)


class TestResolveDLCMConfig(unittest.TestCase):
    def test_preset_spec_returns_config(self) -> None:
        spec = PresetDLCMModelSpec(size="small", max_seq_len=512, target_ratio=4.0)
        config = resolve_dlcm_config(spec)
        self.assertIsInstance(config, DLCMConfig)
        self.assertEqual(config.d_token, 512)
        self.assertEqual(config.d_concept, 1024)
        self.assertEqual(config.max_seq_len, 512)

    def test_tiny_preset(self) -> None:
        config = resolve_dlcm_config(PresetDLCMModelSpec(size="tiny"))
        self.assertEqual(config.d_token, 256)
        self.assertEqual(config.num_backbone_layers, 4)

    def test_custom_spec_returns_config(self) -> None:
        custom = DLCMConfig(d_token=64, d_concept=128)
        config = resolve_dlcm_config(CustomDLCMModelSpec(config=custom))
        self.assertIs(config, custom)

    def test_boundary_mode_propagates(self) -> None:
        config = resolve_dlcm_config(PresetDLCMModelSpec(size="tiny", boundary_mode="rule"))
        self.assertEqual(config.boundary_mode, "rule")


class TestFormatParameterCount(unittest.TestCase):
    def test_millions(self) -> None:
        self.assertEqual(format_parameter_count(4_000_000), "4.0M")

    def test_billions(self) -> None:
        self.assertEqual(format_parameter_count(1_200_000_000), "1.2B")

    def test_small(self) -> None:
        result = format_parameter_count(500)
        self.assertIn("500", result)
