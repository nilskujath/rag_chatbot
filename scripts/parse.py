"""Parse the rulebook PDF into markdown (slow: Docling runs layout models)."""

import argparse
from pathlib import Path

from rag_chatbot.parser import DoclingParser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default="data/raw/core_rules.pdf")
    parser.add_argument("--markdown", type=Path, default="data/processed/core_rules.md")
    args = parser.parse_args()

    markdown = DoclingParser().parse(args.pdf)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown)
    print(f"Wrote {len(markdown)} chars to {args.markdown}")


if __name__ == "__main__":
    main()
