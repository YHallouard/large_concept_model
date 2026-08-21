import unittest

import torch

from lcm_explo.domain.models.nn.segment_pooling._class import (
    SegmentMeanPooling,
    segment_ids_from_boundaries,
)


class TestSegmentIdsFromBoundaries(unittest.TestCase):
    def test_segment_ids_and_counts(self) -> None:
        # Given boundaries for two sequences (b[:, 0] == 1)
        b = torch.tensor(
            [
                [1, 0, 0, 1, 0],  # segments {0,1,2}, {3,4} -> 2 concepts
                [1, 1, 0, 1, 1],  # {0}, {1,2}, {3}, {4}   -> 4 concepts
            ],
            dtype=torch.float32,
        )

        # When
        seg_id, num_segments = segment_ids_from_boundaries(b)

        # Then
        torch.testing.assert_close(seg_id, torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 2, 3]]))
        torch.testing.assert_close(num_segments, torch.tensor([2, 4]))


class TestSegmentMeanPooling(unittest.TestCase):
    def test_pooled_values_match_manual_means(self) -> None:
        # Given a hand-built segmentation and hidden states
        b = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)  # segments {0,1}, {2,3}
        h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [10.0, 0.0], [20.0, 4.0]]])
        seg_id, num_segments = segment_ids_from_boundaries(b)

        # When
        pooled, concept_mask = SegmentMeanPooling()(h, seg_id, num_segments)

        # Then the concepts are the per-segment means
        expected = torch.tensor([[[2.0, 2.0], [15.0, 2.0]]])
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(concept_mask, torch.tensor([[True, True]]))

    def test_concept_mask_marks_padding(self) -> None:
        # Given a batch with a different number of segments per row
        b = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
        h = torch.rand(2, 4, 3)
        seg_id, num_segments = segment_ids_from_boundaries(b)

        # When
        _, concept_mask = SegmentMeanPooling()(h, seg_id, num_segments)

        # Then padding concepts (beyond num_segments) are masked out
        self.assertEqual(concept_mask.shape, (2, 4))
        torch.testing.assert_close(
            concept_mask,
            torch.tensor([[True, False, False, False], [True, True, True, True]]),
        )

    def test_gradient_flows_to_hidden_states(self) -> None:
        # Given
        b = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        h = torch.rand(1, 4, 2, requires_grad=True)
        seg_id, num_segments = segment_ids_from_boundaries(b)

        # When we backprop through the pooled concepts
        pooled, _ = SegmentMeanPooling()(h, seg_id, num_segments)
        pooled.sum().backward()

        # Then the encoder hidden states receive a gradient
        self.assertIsNotNone(h.grad)
        self.assertGreater(h.grad.abs().sum().item(), 0.0)
