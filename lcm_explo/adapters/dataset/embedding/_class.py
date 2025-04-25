from pathlib import Path

import torch
from torch.utils.data import Dataset

from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository


def embedding_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    padding_mask = torch.stack([item[2] for item in batch])
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    inputs = inputs.to(device)
    targets = targets.to(device)
    padding_mask = padding_mask.to(device)
    return inputs, targets, padding_mask


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_dir: Path, sequence_length: int = 32, stride: int = 16):
        """Dataset for training LCM on Wikipedia embeddings.

        Args:
            embeddings_dir: Directory containing the embeddings
            sequence_length: Length of sequences to return
            stride: Stride for sliding window over embeddings
        """
        self.sequence_length = sequence_length
        self.stride = stride

        self.embedding_repo = FileSystemEmbeddingRepository(base_path=embeddings_dir.as_posix())
        self.documents = list(self.embedding_repo.list_documents())

        self.sequence_indices: list[tuple[str, int]] = []
        for doc_id in self.documents:
            doc = self.embedding_repo.load_document_embeddings(doc_id)
            num_sequences = max(1, (len(doc.embeddings) - sequence_length) // self.stride + 1)
            for seq_idx in range(num_sequences):
                self.sequence_indices.append((doc_id, seq_idx * self.stride))

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        doc_id, start_idx = self.sequence_indices[idx]
        doc = self.embedding_repo.load_document_embeddings(doc_id)

        sequence = doc.embeddings[start_idx : start_idx + self.sequence_length + 1]

        input_sequence_part = sequence[:-1]
        input_padding = torch.zeros(
            (self.sequence_length - len(input_sequence_part), sequence.shape[1]), device=sequence.device
        )
        input_sequence = torch.cat([input_sequence_part, input_padding], dim=0)

        target_sequence_part = sequence[1:]
        target_padding = torch.zeros(
            (self.sequence_length - len(target_sequence_part), sequence.shape[1]), device=sequence.device
        )
        target_sequence = torch.cat([target_sequence_part, target_padding], dim=0)

        padding_mask = torch.cat([torch.ones(len(input_sequence_part)), torch.zeros(len(input_padding))], dim=0)

        return input_sequence, target_sequence, padding_mask
