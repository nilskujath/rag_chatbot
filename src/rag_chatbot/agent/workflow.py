"""The RAG agent: a langgraph router over two retrieval channels.

    START -> router -> [rulebook] -> [database] -> final_answer -> END

The router picks `text`, `sql` or `both`; the two retrieval nodes are visited
according to that choice, then one LLM call writes the answer.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_chatbot import llm
from rag_chatbot.agent.nodes import (
    make_final_answer_node,
    make_retrieve_node,
    make_router_node,
)
from rag_chatbot.agent.state import AgentState
from rag_chatbot.retriever import Reranker, Retriever, SqlRetriever, TextRetriever
from rag_chatbot.settings import settings
from rag_chatbot.storage import Embedder, SqliteVecStore

RULEBOOK = "Rulebook"
DATABASE = "Game database"


class SetupError(RuntimeError):
    """A fresh clone is not ready to run; the message says what to do."""


def after_router(state: AgentState) -> str:
    """`sql` goes straight to the database; the other routes start with the rulebook."""
    return "database" if state["route"] == "sql" else "rulebook"


def after_rulebook(state: AgentState) -> str:
    """`both` continues to the database; `text` goes to the answer."""
    return "database" if state["route"] == "both" else "final_answer"


def build_agent(
    generate: Callable[..., str],
    stream: Callable[[str], Iterator[str]],
    text_retriever: Retriever,
    sql_retriever: Retriever,
) -> CompiledStateGraph:
    """Assemble and compile the graph from its components (all injectable)."""
    workflow = StateGraph(AgentState)
    workflow.add_node("router", make_router_node(generate))
    workflow.add_node("rulebook", make_retrieve_node(text_retriever, RULEBOOK))
    workflow.add_node("database", make_retrieve_node(sql_retriever, DATABASE))
    workflow.add_node("final_answer", make_final_answer_node(stream))

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", after_router)
    workflow.add_conditional_edges("rulebook", after_rulebook)
    workflow.add_edge("database", "final_answer")
    workflow.add_edge("final_answer", END)
    return workflow.compile(checkpointer=MemorySaver())


def check_setup(vector_db_path: str | None = None) -> None:
    """Raise `SetupError` with plain instructions when a required piece is missing."""
    problems = []
    if not settings.google_api_key:
        problems.append(
            "GOOGLE_API_KEY is empty: copy .env.example to .env and paste your key."
        )
    if not Path(settings.sql_db_path).exists():
        problems.append(
            f"{settings.sql_db_path} is missing: copy wahadb.sqlite from the course "
            "material (see README, Quick start)."
        )
    if vector_db_path is not None and not Path(vector_db_path).exists():
        problems.append(f"{vector_db_path} is missing: run `make index` first.")
    if problems:
        raise SetupError("\n".join(problems))


def build_default_agent(vector_db_path: str, k: int) -> CompiledStateGraph:
    """Build the agent on the real Gemini client, vector store and database."""
    check_setup(vector_db_path)
    embedder = Embedder()
    store = SqliteVecStore(vector_db_path, embedder.dimension)
    if len(store) == 0:
        raise SetupError(f"{vector_db_path} holds no chunks: run `make index` first.")
    reranker = Reranker() if settings.rerank_candidates else None
    text_retriever = TextRetriever(
        embedder, store, k, reranker, settings.rerank_candidates
    )
    return build_agent(
        llm.generate, llm.stream, text_retriever, SqlRetriever(llm.generate)
    )
