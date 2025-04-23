import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lcm_explo.adapters.dataset.random_sized_embeddings._class import RandomSequenceDataset
from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings


class TestRandomSequenceDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.embeddings_dir = Path(self.temp_dir.name)
        self.min_length = 4
        self.max_length = 32

        # Create mock embeddings
        self.doc1_embeddings = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ])

        # Create documents and save them
        self.embedding_repo = FileSystemEmbeddingRepository(base_path=self.embeddings_dir.as_posix())
        self.doc1 = DocumentEmbeddings(document_id="doc1", embeddings=self.doc1_embeddings)
        self.embedding_repo.save_document_embeddings(self.doc1)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_random_sequence_generation(self) -> None:
        # Given
        torch.manual_seed(0)
        stride = 4
        dataset = RandomSequenceDataset(
            embeddings_dir=self.embeddings_dir, min_length=self.min_length, max_length=self.max_length, stride=stride
        )

        # When
        input_seq, target_seq, padding_mask = dataset[0]

        # Then
        expected_input_seq = torch.cat([self.doc1_embeddings[:-3], torch.zeros((24, 2))], dim=0)
        expected_target_seq = torch.cat([self.doc1_embeddings[1:-2], torch.zeros((24, 2))], dim=0)
        expected_padding_mask = torch.cat([torch.ones(7), torch.zeros(24)])

        torch.testing.assert_close(input_seq, expected_input_seq)
        torch.testing.assert_close(target_seq, expected_target_seq)
        torch.testing.assert_close(padding_mask, expected_padding_mask)
        self.assertEqual(input_seq.shape[:-1], padding_mask.shape)
