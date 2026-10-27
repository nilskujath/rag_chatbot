from abc import ABC, abstractmethod


class Retriever(ABC):
    """Contract for a retrieval channel: a question in, context passages out."""

    @abstractmethod
    def retrieve(self, query: str) -> list[str]:
        """Return the passages relevant to the query, most relevant first."""
