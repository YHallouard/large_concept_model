import unittest

import torch

from lcm_explo.domain.models.dlcm import DLCM, DLCMConfig
from lcm_explo.domain.usecases.generate import _spans_from_boundaries, generate_dlcm


class _StubTokenizer:
    """Minimal char-level tokenizer for testing the generation usecase."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 32) + 1 for c in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i % 32) + 96) for i in ids)


class TestSpansFromBoundaries(unittest.TestCase):
    def test_spans_partition_the_sequence(self) -> None:
        # Given boundaries {0,3} over 5 tokens
        boundaries = torch.tensor([1, 0, 0, 1, 0])

        # When
        spans = _spans_from_boundaries(boundaries)

        # Then the spans tile the sequence without gaps
        self.assertEqual(spans, [(0, 3), (3, 5)])


class TestGenerateDLCM(unittest.TestCase):
    def test_generation_returns_text_and_segments(self) -> None:
        # Given a tiny model and a stub tokenizer
        config = DLCMConfig(
            vocab_size=64,
            d_token=16,
            d_concept=32,
            d_scan=8,
            num_encoder_layers=1,
            num_backbone_layers=1,
            num_decoder_layers=1,
            num_token_heads=2,
            num_concept_heads=2,
            ffn_multiplier=2,
            max_seq_len=32,
        )
        model = DLCM(config)
        model.eval()

        # When
        result = generate_dlcm(model, _StubTokenizer(), "hello", max_new_tokens=4, top_k=5)

        # Then we get decoded text, token ids and non-empty, ordered segments
        self.assertIsInstance(result.text, str)
        self.assertEqual(len(result.token_ids), len("hello") + 4)
        self.assertGreater(len(result.segments), 0)
        for span in result.segments:
            self.assertLess(span.start, span.end)
