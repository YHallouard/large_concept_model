import unittest

import torch

from lcm_explo.domain.models.one_tower_lcm import OneTowerLCM, OneTowerLCMConfig


class TestOneTowerLCM(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = OneTowerLCMConfig(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64, concept_embedding_dim=16
        )
        self.batch_size = 2
        self.seq_length = 8
        self.model_inputs = {
            "input_concepts": torch.randn(self.batch_size, self.seq_length, self.model_config.concept_embedding_dim),
            "target_concepts": torch.randn(self.batch_size, self.seq_length, self.model_config.concept_embedding_dim),
            "noise_level": torch.rand(self.batch_size),
        }

    def test_model_initialization(self) -> None:
        model = OneTowerLCM(self.model_config)

        self.assertIsInstance(model, OneTowerLCM)
        self.assertEqual(model.config, self.model_config)
        self.assertEqual(model.config_class, OneTowerLCMConfig)
        self.assertEqual(model.base_model_prefix, "one_tower_lcm")

    def test_model_forward_train(self) -> None:
        model = OneTowerLCM(self.model_config)
        outputs = model(
            input_concepts=self.model_inputs["input_concepts"],
            target_concepts=self.model_inputs["target_concepts"],
        )

        self.assertIsInstance(outputs, dict)
        self.assertIn("predicted_concepts", outputs)
        self.assertIn("loss", outputs)
        self.assertEqual(outputs["predicted_concepts"].shape, self.model_inputs["target_concepts"].shape)
        self.assertGreater(outputs["loss"].item(), 0)

    def test_model_forward_inference(self) -> None:
        model = OneTowerLCM(self.model_config)
        outputs = model(input_concepts=self.model_inputs["input_concepts"])

        self.assertIsInstance(outputs, dict)
        self.assertIn("predicted_concepts", outputs)
        self.assertNotIn("loss", outputs)
        self.assertEqual(outputs["predicted_concepts"].shape, self.model_inputs["input_concepts"].shape)

    def test_model_forward_with_noise(self) -> None:
        model = OneTowerLCM(self.model_config)
        outputs = model(
            input_concepts=self.model_inputs["input_concepts"],
            target_concepts=self.model_inputs["target_concepts"],
            noise_level=self.model_inputs["noise_level"],
        )

        self.assertIsInstance(outputs, dict)
        self.assertIn("predicted_concepts", outputs)
        self.assertIn("loss", outputs)
        self.assertEqual(outputs["predicted_concepts"].shape, self.model_inputs["target_concepts"].shape)

    def test_model_forward_no_dict(self) -> None:
        model = OneTowerLCM(self.model_config)
        outputs = model(
            input_concepts=self.model_inputs["input_concepts"],
            return_dict=False,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, self.model_inputs["input_concepts"].shape)

    def test_model_gradient_flow(self) -> None:
        model = OneTowerLCM(self.model_config)
        outputs = model(
            input_concepts=self.model_inputs["input_concepts"],
            target_concepts=self.model_inputs["target_concepts"],
        )

        outputs["loss"].backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_cuda(self) -> None:
        model = OneTowerLCM(self.model_config).cuda()
        cuda_inputs = {k: v.cuda() for k, v in self.model_inputs.items()}

        outputs = model(
            input_concepts=cuda_inputs["input_concepts"],
            target_concepts=cuda_inputs["target_concepts"],
        )

        self.assertEqual(outputs["predicted_concepts"].device.type, "cuda")
        self.assertEqual(outputs["loss"].device.type, "cuda")
