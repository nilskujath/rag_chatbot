from abc import ABC, abstractmethod
from pathlib import Path


class Parser(ABC):
    """Contract for PDF to markdown conversion."""

    @abstractmethod
    def parse(self, pdf_path: Path) -> str:
        """Return the document as markdown."""
