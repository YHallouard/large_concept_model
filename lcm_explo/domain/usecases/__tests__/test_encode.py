import unittest

import torch

from lcm_explo.adapters.splitter import InMemorySplitter
from lcm_explo.adapters.tokenizer import InMemoryTokenizer
from lcm_explo.constant import SONAR_DIMENSIONS
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

    def test_split_long_text_with_different_min_lengths(self) -> None:
        # Test with small min length
        result_small = split_long_text(self.tokenizer, self.splitter, self.text, max_length=10, min_sentence_length=5)
        self.assertGreater(len(result_small), 0)

        # Test with large min length
        result_large = split_long_text(self.tokenizer, self.splitter, self.text, max_length=10, min_sentence_length=100)
        self.assertEqual(len(result_large), 0)

    def test_encode_text_sonar(self) -> None:
        # Given
        class MockModel(torch.nn.Module):
            def __init__(self, embedding_dim: int = SONAR_DIMENSIONS) -> None:
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

    def test_encode_text_sonar_with_different_min_lengths(self) -> None:
        # Given
        class MockModel(torch.nn.Module):
            def __init__(self, embedding_dim: int = SONAR_DIMENSIONS) -> None:
                super().__init__()
                self.embedding_dim = embedding_dim

            def forward(self, **kwargs) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
                batch_size = kwargs["input_ids"].shape[0]
                return type("obj", (object,), {"last_hidden_state": torch.ones((batch_size, 5, self.embedding_dim))})  # type: ignore[return-value]

        model = MockModel()

        # Test with small min length
        result_small = encode_text_sonar(
            self.long_text, self.device, self.tokenizer, model, self.splitter, min_sentence_length=5
        )
        self.assertGreater(result_small.shape[0], 99)

        # Test with large min length
        result_large = encode_text_sonar(
            self.long_text, self.device, self.tokenizer, model, self.splitter, min_sentence_length=100
        )
        self.assertEqual(result_large.shape[0], 0)
