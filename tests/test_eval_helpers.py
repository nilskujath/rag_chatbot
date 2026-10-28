import pytest

pytest.importorskip("ragas")

from scripts.chat import turn_record  # noqa: E402
from scripts.eval_retrieval import observed_channel  # noqa: E402


def test_observed_channel_maps_sources_to_the_testset_vocabulary():
    assert observed_channel(["Rulebook"]) == "text"
    assert observed_channel(["Game database"]) == "sql"
    assert observed_channel(["Rulebook", "Game database"]) == "both"
    assert observed_channel([]) == "none"


def test_turn_record_extracts_the_sql_and_rounds_latency():
    state = {
        "route": "sql",
        "query": "q",
        "sources": ["Game database"],
        "contexts": ["##### Game database, passage 1 #####\nSQL query: SELECT 1\nx: 1"],
    }
    record = turn_record("question?", state, 3.14159)
    assert record["sql"] == "SELECT 1"
    assert record["seconds"] == 3.1
    assert record["route"] == "sql" and record["question"] == "question?"


def test_turn_record_without_sql():
    assert (
        turn_record(
            "q",
            {"route": "text", "contexts": ["##### Rulebook, passage 1 #####\ntext"]},
            1,
        )["sql"]
        is None
    )
