from collections.abc import Callable, Iterator

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from rag_chatbot.agent.state import AgentState, history, question

PROMPT = """You are a Warhammer 40,000 rules assistant. Answer the question from the \
context below. Quote the relevant rule text or figures (never the passage headers). \
Database rows are the result of a SQL query written for the question, so they \
already match its conditions; when there are many rows, give the typical values \
and the range rather than the full list. The rows are complete unless a passage \
ends with "(only the first N rows are shown)"; only then say the list is partial. \
If the context does not cover the question, say what is missing rather than guess.

## Conversation so far
{history}

## Context (from: {sources})
{context}

## Question
{question}

## Answer"""


def make_final_answer_node(stream: Callable[[str], Iterator[str]]):
    """Create the node that writes the answer, token by token, from the contexts."""

    def final_answer_node(state: AgentState) -> dict:
        prompt = PROMPT.format(
            history=history(state),
            sources=", ".join(state["sources"]) or "nothing retrieved",
            context="\n\n".join(state["contexts"]) or "No context found.",
            question=question(state),
        )
        write = get_stream_writer()
        tokens = []
        for token in stream(prompt):
            write(token)
            tokens.append(token)
        return {"messages": [AIMessage(content="".join(tokens))]}

    return final_answer_node
