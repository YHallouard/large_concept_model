import torch
from torch.utils.data import Dataset

from lcm_explo.domain.models.documents import DocumentEmbeddings


class InMemoryEmbeddingsDataset(Dataset):
    """In-memory implementation of EmbeddingsDataset for testing purposes."""

    def __init__(self, documents: list[DocumentEmbeddings], sequence_length: int = 32, stride: int = 16) -> None:
        """Initialize the dataset.

        Args:
            documents: List of DocumentEmbeddings to use
            sequence_length: Length of sequences to return
            stride: Stride for sliding window over embeddings
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.documents = documents

        self.sequence_indices: list[tuple[int, int]] = []
        for doc_idx, doc in enumerate(documents):
            num_sequences = max(1, (len(doc.embeddings) - sequence_length) // self.stride + 1)
            for seq_idx in range(num_sequences):
                self.sequence_indices.append((doc_idx, seq_idx * self.stride))

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        doc_idx, start_idx = self.sequence_indices[idx]
        doc = self.documents[doc_idx]

        sequence = doc.embeddings[start_idx : start_idx + self.sequence_length]
        if len(sequence) < self.sequence_length:
            padding = torch.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = torch.cat([sequence, padding], dim=0)

        padding_mask = torch.ones(sequence.shape[:-1])

        return sequence[:-1], sequence[1:], padding_mask[:-1]
