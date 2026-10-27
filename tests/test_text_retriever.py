from unittest.mock import MagicMock

from rag_chatbot.chunker import Chunk
from rag_chatbot.retriever import TextRetriever

CHUNKS = [Chunk(f"text {i}", f"section {i}") for i in range(4)]


def store_with(chunks):
    store = MagicMock()
    store.search.side_effect = lambda embedding, k: chunks[:k]
    return store


def embedder():
    fake = MagicMock()
    fake.embed_query.return_value = [0.0]
    return fake


def test_without_reranker_returns_k_nearest_in_store_order():
    store = store_with(CHUNKS)
    passages = TextRetriever(embedder(), store, k=2).retrieve("q")
    store.search.assert_called_once_with([0.0], 2)
    assert passages == ["section: section 0\ntext 0", "section: section 1\ntext 1"]


def test_with_reranker_scores_candidates_and_keeps_k():
    store = store_with(CHUNKS)
    reranker = MagicMock()
    reranker.top.side_effect = lambda query, chunks, k: list(reversed(chunks))[:k]
    retriever = TextRetriever(embedder(), store, k=2, reranker=reranker, candidates=4)
    passages = retriever.retrieve("q")
    store.search.assert_called_once_with([0.0], 4)
    reranker.top.assert_called_once_with("q", CHUNKS, 2)
    assert passages == ["section: section 3\ntext 3", "section: section 2\ntext 2"]


def test_candidates_never_below_k():
    retriever = TextRetriever(embedder(), store_with(CHUNKS), k=3, candidates=1)
    assert retriever.candidates == 3
