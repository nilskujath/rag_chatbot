from pathlib import Path

from docling.document_converter import DocumentConverter

from rag_chatbot.parser.base import Parser


class DoclingParser(Parser):
    """Layout-aware parser: Docling detects headers, tables and lists."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, pdf_path: Path) -> str:
        result = self.converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
