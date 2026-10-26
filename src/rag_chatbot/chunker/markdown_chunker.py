from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_chatbot.chunker.base import Chunk, Chunker
from rag_chatbot.settings import settings


class MarkdownChunker(Chunker):
    """Split on markdown headers first, then on length inside each section.

    The header path (for example `Shooting phase > Hit roll`) is kept on every
    chunk so that the context handed to the LLM can name its section.
    Chunks shorter than `min_length` (stray headers, table fragments) are dropped.
    Parameters left at `None` come from the settings.
    """

    def __init__(
        self,
        max_length: int | None = None,
        overlap: int | None = None,
        min_length: int | None = None,
    ):
        if max_length is None:
            max_length = settings.chunk_max_length
        if overlap is None:
            overlap = settings.chunk_overlap
        if min_length is None:
            min_length = settings.chunk_min_length
        self.min_length = min_length
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=settings.chunk_headers_to_split,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=overlap,
            separators=settings.chunk_separators,
        )

    def chunk(self, text: str) -> list[Chunk]:
        sections = self.header_splitter.split_text(text)
        documents = self.text_splitter.split_documents(sections)
        return [
            Chunk(doc.page_content, " > ".join(doc.metadata.values()))
            for doc in documents
            if len(doc.page_content) >= self.min_length
        ]
