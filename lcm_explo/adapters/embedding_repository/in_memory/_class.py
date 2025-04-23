from lcm_explo.domain.infrastructures.embedding_repository import EmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings


class InMemoryEmbeddingRepository(EmbeddingRepository):
    """In-memory implementation of EmbeddingRepository for testing."""

    def __init__(self) -> None:
        self._embeddings: dict[str, DocumentEmbeddings] = {}

    def save_document_embeddings(self, embeddings: DocumentEmbeddings) -> None:
        """Save embeddings to in-memory storage."""
        self._embeddings[embeddings.document_id] = embeddings

    def list_documents(self) -> list[str]:
        """List all document IDs in the repository."""
        return list(self._embeddings.keys())

    def load_document_embeddings(self, document_id: str) -> DocumentEmbeddings:
        """Load embeddings for a document."""
        return self._embeddings[document_id]
