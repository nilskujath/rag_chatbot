from sentence_transformers import SentenceTransformer

from rag_chatbot.settings import settings


class Embedder:
    """Sentence-transformer model that maps texts to vectors."""

    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    @property
    def dimension(self) -> int:
        """Length of the vectors, needed to declare the vector table."""
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"{self.model} does not report an embedding dimension")
        return dimension

    def embed_query(self, text: str) -> list[float]:
        """Vector of one query."""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectors of many passages, in one batch."""
        return self.model.encode(texts).tolist()
