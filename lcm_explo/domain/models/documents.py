from dataclasses import dataclass

import torch


@dataclass
class DocumentEmbeddings:
    """Represents all embeddings for a document."""

    document_id: str
    embeddings: torch.Tensor  # Shape: (num_sentences, embedding_dim)
