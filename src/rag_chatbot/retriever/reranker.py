from sentence_transformers import CrossEncoder

from rag_chatbot.chunker.base import Chunk
from rag_chatbot.settings import settings


class Reranker:
    """Cross-encoder that reads query and chunk together and scores the pair.

    Slower than a vector comparison, so it runs on a short candidate list, but
    more precise: it sees the words of both texts at once.
    """

    def __init__(self, model_name: str | None = None):
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def top(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Return the k chunks the cross-encoder scores highest, best first.

        An empty candidate list gives an empty result without a model call.
        """
        if not chunks:
            return []
        scores = self.model.predict([(query, chunk.content) for chunk in chunks])
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:k]]
