import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lcm_explo.adapters.dataset.embedding._class import EmbeddingsDataset
from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings


class TestEmbeddingsDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.embeddings_dir = Path(self.temp_dir.name)
        self.sequence_length = 4
        self.stride = 2

        # Create mock embeddings
        self.doc1_embeddings = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ])
        self.doc2_embeddings = torch.tensor([
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
        ])

        # Create documents and save them
        self.embedding_repo = FileSystemEmbeddingRepository(base_path=self.embeddings_dir.as_posix())
        self.doc1 = DocumentEmbeddings(document_id="doc1", embeddings=self.doc1_embeddings)
        self.doc2 = DocumentEmbeddings(document_id="doc2", embeddings=self.doc2_embeddings)
        self.embedding_repo.save_document_embeddings(self.doc1)
        self.embedding_repo.save_document_embeddings(self.doc2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init_and_len(self) -> None:
        # Given
        # - embeddings_dir with two documents saved
        # - sequence_length = 4
        # - stride = 2

        # When
        dataset = EmbeddingsDataset(
            embeddings_dir=self.embeddings_dir, sequence_length=self.sequence_length, stride=self.stride
        )

        # Then
        self.assertEqual(dataset.sequence_length, self.sequence_length)
        self.assertEqual(dataset.stride, self.stride)
        self.assertEqual(set(dataset.documents), {"doc1", "doc2"})

        # For doc1: (5 - 4) // 2 + 1 = 1 sequence
        # For doc2: max(1, (3 - 4) // 2 + 1) = 1 sequence
        expected_len = 2  # Total sequences across all documents
        self.assertEqual(len(dataset), expected_len)

    def test_getitem(self) -> None:
        # Given
        # - embeddings_dir with doc1 saved
        # - sequence_length = 4
        # - stride = 2

        # When
        dataset = EmbeddingsDataset(
            embeddings_dir=self.embeddings_dir, sequence_length=self.sequence_length, stride=self.stride
        )
        input_seq, target_seq, padding_mask = dataset[0]

        # Then
        expected_input = self.doc1_embeddings[:3]  # First 3 elements
        expected_target = self.doc1_embeddings[1:4]  # Elements 1-4
        expected_padding_mask = torch.ones(expected_input.shape[:-1])

        torch.testing.assert_close(input_seq, expected_input)
        torch.testing.assert_close(target_seq, expected_target)
        torch.testing.assert_close(padding_mask, expected_padding_mask)

    def test_getitem_with_padding(self) -> None:
        # Given
        # - embeddings_dir with only doc2 saved
        # - sequence_length = 4
        # - stride = 2
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            embedding_repo = FileSystemEmbeddingRepository(base_path=tmp_path.as_posix())
            embedding_repo.save_document_embeddings(self.doc2)

            # When
            dataset = EmbeddingsDataset(
                embeddings_dir=tmp_path, sequence_length=self.sequence_length, stride=self.stride
            )
            input_seq, target_seq, padding_mask = dataset[0]

            # Then
            padding = torch.zeros((1, 2))  # 1 padding element with 2 features
            expected_input = torch.cat([self.doc2_embeddings[:3], padding])[:-1]
            expected_target = torch.cat([self.doc2_embeddings[:3], padding])[1:]
            expected_padding_mask = torch.ones(expected_input.shape[:-1])

            torch.testing.assert_close(input_seq, expected_input)
            torch.testing.assert_close(target_seq, expected_target)
            torch.testing.assert_close(padding_mask, expected_padding_mask)
