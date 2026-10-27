from rag_chatbot.agent.state import AgentState
from rag_chatbot.retriever.base import Retriever


def make_retrieve_node(retriever: Retriever, source: str):
    """Create a node that appends one channel's passages to the turn's contexts."""

    def retrieve_node(state: AgentState) -> dict:
        passages = retriever.retrieve(state["query"])
        offset = len(state["contexts"])
        contexts = [
            f"##### {source}, passage {offset + i} #####\n{passage}"
            for i, passage in enumerate(passages, 1)
        ]
        return {
            "contexts": state["contexts"] + contexts,
            "sources": state["sources"] + [source],
        }

    return retrieve_node
