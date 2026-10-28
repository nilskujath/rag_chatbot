"""Helpers shared by the entry points: logging, conversation threads, JSONL files."""

import json
import logging
import sys
import uuid
from pathlib import Path

from rag_chatbot.agent import SetupError, build_default_agent


def configure_logging(handler: logging.Handler | None = None) -> logging.Logger:
    """Show the project's trace at INFO, keep third-party libraries at WARNING."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[handler] if handler else None,
    )
    trace = logging.getLogger("rag_chatbot")
    trace.setLevel(logging.INFO)
    return trace


def new_thread() -> dict:
    """A fresh langgraph thread id: one per conversation (chat) or question (eval)."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def agent_or_exit(vector_db_path: str, k: int):
    """Build the default agent, or print what is missing and exit with status 1."""
    try:
        return build_default_agent(vector_db_path, k)
    except SetupError as error:
        sys.exit(f"Not ready to run:\n{error}")


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file; [] when the file does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON lines, one per row, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    )


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row to a JSON-lines file, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
