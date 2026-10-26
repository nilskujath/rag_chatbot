"""Chunk the parsed markdown, embed the chunks and (re)build the vector store."""

import argparse
from pathlib import Path

from rag_chatbot.chunker import MarkdownChunker
from rag_chatbot.settings import settings
from rag_chatbot.storage import Embedder, SqliteVecStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default="data/processed/core_rules.md")
    parser.add_argument("--vector-db", default=settings.vector_db_path)
    args = parser.parse_args()

    chunks = MarkdownChunker().chunk(args.markdown.read_text())
    print(f"{len(chunks)} chunks")
    embedder = Embedder()
    embeddings = embedder.embed_documents([chunk.content for chunk in chunks])
    store = SqliteVecStore(args.vector_db, embedder.dimension)
    store.replace(chunks, embeddings)
    print(f"Stored {len(store)} chunks in {args.vector_db}")


if __name__ == "__main__":
    main()
