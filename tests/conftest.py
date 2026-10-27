"""Shared fixtures: fake LLM callables and in-memory retrievers, no network."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from rag_chatbot.retriever.base import Retriever


class FakeRetriever(Retriever):
    """Returns fixed passages and remembers the queries it received."""

    def __init__(self, passages: list[str]):
        self.passages = passages
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        return list(self.passages)


@pytest.fixture()
def fake_generate():
    """LLM stand-in: routes to `text`, rewrites the query, writes `SELECT 1`."""

    def generate(prompt: str, json_output: bool = False, temperature=None) -> str:
        if json_output:
            return json.dumps({"route": "text", "query": "rewritten query"})
        return "SELECT 1"

    return MagicMock(side_effect=generate)


@pytest.fixture()
def fake_stream():
    """Streams a fixed answer and remembers every prompt."""
    return MagicMock(side_effect=lambda prompt: iter(["The ", "answer."]))


@pytest.fixture()
def rulebook():
    return FakeRetriever(["section: Shooting\nRoll to hit."])


@pytest.fixture()
def database():
    return FakeRetriever(["name: Intercessors, T: 4"])


@pytest.fixture()
def units_db(tmp_path):
    """A tiny SQLite file with one table, plus a one-line data dictionary."""
    db_path = tmp_path / "units.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE units (name TEXT, toughness INTEGER)")
        conn.executemany(
            "INSERT INTO units VALUES (?, ?)", [("Intercessors", 4), ("Terminators", 5)]
        )
    dictionary = tmp_path / "dictionary.md"
    dictionary.write_text("units(name TEXT, toughness INTEGER)")
    return str(db_path), str(dictionary)
