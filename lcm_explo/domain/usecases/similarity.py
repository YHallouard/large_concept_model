from typing import cast

import numpy as np
from numpy.typing import NDArray


def normalize_embeddings(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize embeddings to unit length.

    Args:
        embeddings: Input embeddings to normalize

    Returns:
        Normalized embeddings
    """
    norm_result = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return cast(NDArray[np.float64], norm_result)


def compute_similarities(source_embs: NDArray[np.float64], target_embs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute cosine similarity matrix between source and target embeddings.

    Args:
        source_embs: Source embeddings
        target_embs: Target embeddings

    Returns:
        Similarity matrix
    """
    src_norm = normalize_embeddings(source_embs)
    tgt_norm = normalize_embeddings(target_embs)
    sim_result = np.dot(src_norm, tgt_norm.T)
    return cast(NDArray[np.float64], sim_result)


def compute_xsim(source_embs: NDArray[np.float64], target_embs: NDArray[np.float64], k: int = 4) -> float:
    """Compute cross-similarity error rate between source and target embeddings.

    Args:
        source_embs: Source embeddings
        target_embs: Target embeddings
        k: Number of top matches to consider

    Returns:
        Cross-similarity error rate
    """
    sim_matrix = compute_similarities(source_embs, target_embs)
    n = source_embs.shape[0]

    # Find top-k matches
    top_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]
    top_values = np.take_along_axis(sim_matrix, top_indices, axis=1)

    # Sort top-k results
    sorted_indices = np.argsort(-top_values, axis=1)
    top_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)

    # Calculate error rate
    correct = np.sum(top_indices[:, 0] == np.arange(n))
    error_rate = float(1.0 - (correct / n))
    return error_rate
