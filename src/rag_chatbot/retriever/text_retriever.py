from rag_chatbot.retriever.base import Retriever
from rag_chatbot.retriever.reranker import Reranker
from rag_chatbot.storage.embedder import Embedder
from rag_chatbot.storage.vector_store import SqliteVecStore


class TextRetriever(Retriever):
    """Semantic search over the rulebook chunks, with an optional rerank step.

    Without a reranker the k nearest chunks are returned. With one, `candidates`
    chunks come out of the vector store and the reranker keeps the best k.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: SqliteVecStore,
        k: int,
        reranker: Reranker | None = None,
        candidates: int = 0,
    ):
        self.embedder = embedder
        self.store = store
        self.k = k
        self.reranker = reranker
        self.candidates = max(candidates, k)

    def retrieve(self, query: str) -> list[str]:
        query_embedding = self.embedder.embed_query(query)
        if self.reranker is None:
            chunks = self.store.search(query_embedding, self.k)
        else:
            candidates = self.store.search(query_embedding, self.candidates)
            chunks = self.reranker.top(query, candidates, self.k)
        return [f"section: {chunk.section}\n{chunk.content}" for chunk in chunks]
