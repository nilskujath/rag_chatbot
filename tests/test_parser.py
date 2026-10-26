from pathlib import Path
from unittest.mock import patch

from rag_chatbot.parser import DoclingParser


def test_docling_parser_exports_markdown():
    with patch("rag_chatbot.parser.docling_parser.DocumentConverter") as converter:
        document = converter.return_value.convert.return_value.document
        document.export_to_markdown.return_value = "# Hello"
        assert DoclingParser().parse(Path("dummy.pdf")) == "# Hello"
        converter.return_value.convert.assert_called_once_with("dummy.pdf")
