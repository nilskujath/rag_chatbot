from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from rag_chatbot.settings import settings


class AgentState(TypedDict):
    """State carried along a conversation.

    `messages` accumulates across turns (that is the chat memory); the other
    fields describe the current turn only and are reset by the router.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    query: str
    contexts: list[str]
    sources: list[str]


def question(state: AgentState) -> str:
    """Content of the latest user message."""
    return state["messages"][-1].content


def history(state: AgentState) -> str:
    """Earlier turns of the conversation as `User:` / `Assistant:` lines."""
    earlier = state["messages"][:-1][-2 * settings.history_turns :]
    lines = [
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in earlier
    ]
    return "\n".join(lines) or "(start of conversation)"
