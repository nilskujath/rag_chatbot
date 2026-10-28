"""Generate a RAGAS test set (question, reference answer, source passages) from the
parsed rulebook. Review the rows by hand, drop weak ones, then commit the file:
a frozen test set is what makes two evaluation runs comparable."""

import argparse
import random
import re
from pathlib import Path

from langchain_core.documents import Document
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from rag_chatbot.evaluate import EmbedderRagasEmbeddings, GeminiRagasLLM
from rag_chatbot.storage import Embedder
from scripts._common import configure_logging, write_jsonl


def sections(markdown: str, min_chars: int) -> list[Document]:
    """One document per `## ` section with enough text to carry a question."""
    documents = []
    for block in re.split(r"\n(?=## )", markdown):
        title, _, body = block.partition("\n")
        if len(body) >= min_chars:
            documents.append(
                Document(page_content=block, metadata={"section": title.strip("# ")})
            )
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default="data/processed/core_rules.md")
    parser.add_argument("--output", type=Path, default="data/eval/testset.jsonl")
    parser.add_argument("--size", type=int, default=15, help="number of questions")
    parser.add_argument("--max-sections", type=int, default=25, help="LLM budget")
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    configure_logging()

    documents = sections(args.markdown.read_text(), args.min_chars)
    random.Random(args.seed).shuffle(documents)
    documents = documents[: args.max_sections]
    print(f"Source sections: {len(documents)}")

    generator = TestsetGenerator(
        llm=GeminiRagasLLM(),
        embedding_model=EmbedderRagasEmbeddings(Embedder()),
    )
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=args.size,
        run_config=RunConfig(max_workers=1, timeout=600),
        raise_exceptions=False,
    )

    rows = testset.to_pandas().to_dict(orient="records")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}; review them before you commit.")


if __name__ == "__main__":
    main()
