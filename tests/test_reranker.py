from unittest.mock import patch

from rag_chatbot.chunker import Chunk
from rag_chatbot.retriever import Reranker

CHUNKS = [Chunk("a", "s"), Chunk("b", "s"), Chunk("c", "s")]


def reranker_with_scores(scores):
    with patch("rag_chatbot.retriever.reranker.CrossEncoder") as cross_encoder:
        cross_encoder.return_value.predict.return_value = scores
        return Reranker("any-model")


def test_top_orders_by_score_and_keeps_k():
    reranker = reranker_with_scores([0.1, 0.9, 0.5])
    assert reranker.top("q", CHUNKS, 2) == [CHUNKS[1], CHUNKS[2]]
    pairs = reranker.model.predict.call_args.args[0]
    assert pairs == [("q", "a"), ("q", "b"), ("q", "c")]


def test_top_on_no_candidates_skips_the_model():
    reranker = reranker_with_scores([])
    assert reranker.top("q", [], 5) == []
    reranker.model.predict.assert_not_called()
