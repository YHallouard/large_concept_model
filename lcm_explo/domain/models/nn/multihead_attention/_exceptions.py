class EmbeddingDimensionDefinitionError(Exception):
    """Exception raised for errors in the embedding dimension."""

    def __init__(self, message: str = "Embedding dimension must be 0 modulo number of heads") -> None:
        self.message = message
        super().__init__(self.message)


class EmbeddingDimensionMismatchError(Exception):
    """Exception raised for errors in the embedding dimension."""

    def __init__(self, message: str = "Query embedding dimension must match layer embedding dimension") -> None:
        self.message = message
        super().__init__(self.message)
