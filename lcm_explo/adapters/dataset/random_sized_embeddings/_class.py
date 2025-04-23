import secrets
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository


class RandomSequenceDataset(Dataset):
    def __init__(self, embeddings_dir: Path, min_length: int = 8, max_length: int = 32, stride: int = 16):
        """Dataset for training LCM on Wikipedia embeddings with random sequence lengths.

        Args:
            embeddings_dir: Directory containing the embeddings
            min_length: Minimum length of sequences to return
            max_length: Maximum length of sequences to return
            stride: Stride for sliding window over embeddings
        """
        self.min_length = min_length
        self.max_length = max_length
        self.stride = stride

        self.embedding_repo = FileSystemEmbeddingRepository(base_path=embeddings_dir.as_posix())
        self.documents = list(self.embedding_repo.list_documents())

        self.sequence_indices: list[tuple[str, int, int]] = []
        for doc_id in self.documents:
            doc = self.embedding_repo.load_document_embeddings(doc_id)
            doc_length = len(doc.embeddings)
            if doc_length >= self.min_length:
                start_idx = 0
                while start_idx < doc_length - self.min_length + 1:
                    max_length = min(self.max_length, doc_length - start_idx)
                    sequence_length = secrets.randbelow(max_length - self.min_length + 1) + self.min_length
                    self.sequence_indices.append((doc_id, start_idx, sequence_length))
                    start_idx += sequence_length - self.stride

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        doc_id, start_idx, sequence_length = self.sequence_indices[idx]
        doc = self.embedding_repo.load_document_embeddings(doc_id)

        input_sequence = doc.embeddings[start_idx : start_idx + sequence_length - 1]
        target_sequence = doc.embeddings[start_idx + 1 : start_idx + sequence_length]
        padding_length = self.max_length - sequence_length
        if padding_length > 0:
            input_padding = torch.zeros((padding_length, input_sequence.shape[1]), device=input_sequence.device)
            input_sequence = torch.cat([input_sequence, input_padding], dim=0)

            target_padding = torch.zeros((padding_length, target_sequence.shape[1]), device=target_sequence.device)
            target_sequence = torch.cat([target_sequence, target_padding], dim=0)

        padding_mask = torch.cat([torch.ones(sequence_length - 1), torch.zeros(padding_length)], dim=0)

        return input_sequence, target_sequence, padding_mask
