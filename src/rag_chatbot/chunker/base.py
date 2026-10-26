from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A passage of the source document and the section path it comes from."""

    content: str
    section: str


class Chunker(ABC):
    """Contract for text to chunks conversion."""

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split a document into chunks."""
