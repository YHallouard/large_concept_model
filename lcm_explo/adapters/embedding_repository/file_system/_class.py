from pathlib import Path

import torch

from lcm_explo.adapters.embedding_repository.file_system._exceptions import EmbeddingNotFoundError
from lcm_explo.domain.infrastructures.embedding_repository import EmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings


class FileSystemEmbeddingRepository(EmbeddingRepository):
    """File system implementation of EmbeddingRepository."""

    def __init__(self, base_path: str) -> None:
        """Initialize the repository.

        Args:
            base_path: Base directory to store embeddings
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.embeddings_dir = self.base_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

    def _get_embedding_path(self, document_id: str) -> Path:
        embedding_path = self.embeddings_dir / f"{document_id}.pt"
        return embedding_path

    def save_document_embeddings(self, embeddings: DocumentEmbeddings) -> None:
        embedding_path = self._get_embedding_path(embeddings.document_id)
        torch.save(embeddings.embeddings, embedding_path)

    def list_documents(self) -> list[str]:
        """List all document IDs in the repository.

        Returns:
            List of document IDs
        """
        return [path.stem for path in self.embeddings_dir.glob("*.pt")]

    def load_document_embeddings(self, document_id: str) -> DocumentEmbeddings:
        """Load embeddings for a document.

        Args:
            document_id: ID of the document to load

        Returns:
            DocumentEmbeddings object containing the document's embeddings
        """
        embedding_path = self._get_embedding_path(document_id)
        if not embedding_path.exists():
            raise EmbeddingNotFoundError(document_id)

        embeddings = torch.load(embedding_path)  # nosec
        return DocumentEmbeddings(document_id=document_id, embeddings=embeddings)
