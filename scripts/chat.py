"""Interactive chat with the RAG agent. Answers stream; the trace shows the route,
the generated SQL and the LLM latencies."""

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from google.genai import errors
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown

from rag_chatbot.settings import settings
from scripts._common import agent_or_exit, append_jsonl, configure_logging, new_thread

COMMANDS = {"/reset": "new conversation", "/trace": "toggle the trace", "/quit": "quit"}
LOG = Path("data/logs/chat.jsonl")


def turn_record(question: str, state: dict, seconds: float) -> dict:
    """What a turn did, for the per-turn log: route, query, sources, SQL, latency."""
    sql = next(
        (
            c.split("\n", 2)[1][len("SQL query: ") :]
            for c in state.get("contexts", [])
            if "\nSQL query: " in c
        ),
        None,
    )
    return {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": question,
        "route": state.get("route"),
        "query": state.get("query"),
        "sources": state.get("sources"),
        "sql": sql,
        "seconds": round(seconds, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-db", default=settings.vector_db_path)
    parser.add_argument("-k", type=int, default=settings.vector_k)
    args = parser.parse_args()

    console = Console()
    trace = configure_logging(
        RichHandler(console=console, show_path=False, show_time=False)
    )
    agent = agent_or_exit(args.vector_db, args.k)
    session = PromptSession()
    thread = new_thread()

    console.print("[bold]Warhammer 40k rules assistant[/bold]")
    console.print(
        "[dim]" + "   ".join(f"{k} {v}" for k, v in COMMANDS.items()) + "[/dim]\n"
    )

    while True:
        try:
            text = session.prompt(HTML("<ansigreen><b>&gt;</b></ansigreen> ")).strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            thread = new_thread()
            console.print("[dim]new conversation[/dim]")
            continue
        if text == "/trace":
            turn_off = trace.level == logging.INFO
            trace.setLevel(logging.WARNING if turn_off else logging.INFO)
            console.print(f"[dim]trace {'off' if turn_off else 'on'}[/dim]")
            continue

        answer, state, start = "", {}, time.perf_counter()
        try:
            with Live(console=console, refresh_per_second=12) as live:
                for mode, payload in agent.stream(
                    {"messages": [("user", text)]},
                    thread,
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        answer += payload
                        live.update(Markdown(answer))
                    else:
                        state = payload
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/yellow]")
            continue
        except errors.APIError as error:
            console.print(f"[red]Gemini error {error.code}:[/red] {error.message}")
            continue
        console.print(f"[dim]sources: {', '.join(state.get('sources', []))}[/dim]\n")
        append_jsonl(LOG, turn_record(text, state, time.perf_counter() - start))


if __name__ == "__main__":
    main()
