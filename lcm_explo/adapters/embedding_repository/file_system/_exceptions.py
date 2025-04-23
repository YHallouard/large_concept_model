class EmbeddingNotFoundError(Exception):
    def __init__(self, document_id: str) -> None:
        self.message = f"Embedding not found for document {document_id}"
        super().__init__(self.message)
