import unittest

import torch

from lcm_explo.adapters.splitter import InMemorySplitter
from lcm_explo.adapters.tokenizer import InMemoryTokenizer
from lcm_explo.domain.usecases.encode import encode_text_sonar, split_long_text


class TestEncode(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "This is a test text, it should be longer than the minimum chunk size. . Arf. This is another sentence that is longer than the minimum chunk size."
        self.splitted_text = [
            "This is a test text, it should be longer than the minimum chunk size.",
            " ",
            "Arf.",
            "This is another sentence that is longer than the minimum chunk size.",
        ]
        self.long_text = self.text * 50
        self.splitted_long_text = [
            "This is a test text, it should be longer than the minimum chunk size.",
            " ",
            "Arf.",
            "This is another sentence that is longer than the minimum chunk size.",
        ] * 50
        self.tokenizer = InMemoryTokenizer(vocab_size=10)
        self.splitter = InMemorySplitter({
            self.text: self.splitted_text,
            self.long_text: self.splitted_long_text,
        })
        self.device = "cpu"

    def test_split_long_text_short_input(self) -> None:
        # Given
        short_text = "Short text"

        # When
        result = split_long_text(self.tokenizer, self.splitter, short_text, max_length=20)

        # Assert
        self.assertEqual(result, [short_text])

    def test_split_long_text_long_input(self) -> None:
        # When
        result = split_long_text(self.tokenizer, self.splitter, self.text, max_length=10)

        # Then
        expected_chunks = [
            "This is a test text, it should be longer than the minimum chunk size.",
            "This is another sentence that is longer than the minimum chunk size.",
        ]
        self.assertEqual(result, expected_chunks)

    def test_encode_text_sonar(self) -> None:
        # Given
        class MockModel(torch.nn.Module):
            def __init__(self, embedding_dim: int = 768) -> None:
                super().__init__()
                self.embedding_dim = embedding_dim

            def forward(self, **kwargs) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
                batch_size = kwargs["input_ids"].shape[0]
                return type("obj", (object,), {"last_hidden_state": torch.ones((batch_size, 5, self.embedding_dim))})  # type: ignore[return-value]

        model = MockModel()

        # When
        result = encode_text_sonar(self.long_text, self.device, self.tokenizer, model, self.splitter)

        # Then
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 768)
