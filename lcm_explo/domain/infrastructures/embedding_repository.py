from abc import ABC, abstractmethod

from lcm_explo.domain.models.documents import DocumentEmbeddings


class EmbeddingRepository(ABC):
    """Abstract base class for embedding repositories."""

    @abstractmethod
    def save_document_embeddings(self, embeddings: DocumentEmbeddings) -> None:
        """Save embeddings for a document.

        Args:
            embeddings: DocumentEmbeddings object containing sentences and their embeddings
        """
        pass

    @abstractmethod
    def list_documents(self) -> list[str]:
        """List all document IDs in the repository.

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def load_document_embeddings(self, document_id: str) -> DocumentEmbeddings:
        """Load embeddings for a document.

        Args:
            document_id: ID of the document to load

        Returns:
            DocumentEmbeddings object containing the document's embeddings
        """
        pass
