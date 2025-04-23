import unittest

import numpy as np

from lcm_explo.domain.usecases.similarity import compute_similarities, compute_xsim, normalize_embeddings


class TestSimilarity(unittest.TestCase):
    def test_normalize_embeddings(self) -> None:
        """Test that embeddings normalization works correctly."""
        # Create test embeddings
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

        # Normalize embeddings
        normalized = normalize_embeddings(embeddings)

        # Check shape
        self.assertEqual(normalized.shape, embeddings.shape)

        # Check norms (should be 1)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_almost_equal(norms, np.ones(embeddings.shape[0]))

    def test_compute_similarities(self) -> None:
        """Test that similarity computation works correctly."""
        # Create test embeddings
        source_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        target_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        # Compute similarities
        similarities = compute_similarities(source_embs, target_embs)

        # Expected similarity matrix (identity matrix for orthogonal unit vectors)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Check result
        np.testing.assert_almost_equal(similarities, expected)

    def test_compute_xsim(self) -> None:
        """Test that cross-similarity computation works correctly."""
        # Create test embeddings with perfect diagonal similarity
        n = 10
        source_embs = np.eye(n, dtype=np.float64)
        target_embs = np.eye(n, dtype=np.float64)

        # Compute xsim with default parameters
        error_rate = compute_xsim(source_embs, target_embs)

        # For perfect diagonal similarity, error rate should be 0
        self.assertEqual(error_rate, 0.0)

        # Test with shuffled target embeddings
        shuffled_target = np.roll(target_embs, 1, axis=0)
        error_rate_shuffled = compute_xsim(source_embs, shuffled_target)

        # All matches should be wrong now
        self.assertEqual(error_rate_shuffled, 1.0)
