"""Run the text-to-SQL channel on its own: the generated query and its rows."""

import argparse
import sys

from google.genai import errors

from rag_chatbot.agent import SetupError, check_setup
from rag_chatbot.llm import generate
from rag_chatbot.retriever import SqlRetriever
from scripts._common import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    configure_logging()

    try:
        check_setup()
        [passage] = SqlRetriever(generate).retrieve(args.question)
    except SetupError as error:
        sys.exit(f"Not ready to run:\n{error}")
    except errors.APIError as error:
        sys.exit(f"Gemini error {error.code}: {error.message}")
    print(passage)


if __name__ == "__main__":
    main()
