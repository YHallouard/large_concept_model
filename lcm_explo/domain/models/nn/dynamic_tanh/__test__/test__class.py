import unittest

import torch
from torch.nn.functional import tanh

from lcm_explo.domain.models.nn.dynamic_tanh._class import DyT


class TestDyT(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 4
        self.init_alpha = 0.5
        self.model = DyT(embed_dim=self.embed_dim, init_alpha=self.init_alpha)

    def test_forward(self) -> None:
        # Given
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])

        # When
        output = self.model(x)

        # Then
        expected_output = self.model.gamma * tanh(self.model.alpha * x) + self.model.beta
        torch.testing.assert_close(output, expected_output)

    def test_parameters_initialization(self) -> None:
        self.assertEqual(self.model.alpha.item(), self.init_alpha)
        torch.testing.assert_close(self.model.gamma, torch.ones(self.embed_dim))
        torch.testing.assert_close(self.model.beta, torch.zeros(self.embed_dim))
