import json
import logging
from collections.abc import Callable

from rag_chatbot.agent.state import AgentState, history, question

logger = logging.getLogger(__name__)

ROUTES = ("text", "sql", "both")

PROMPT = """You route questions for a Warhammer 40,000 assistant that has two sources:
- "text": the core rulebook (rules, phases, mechanics, stratagems, missions, \
Victory points)
- "sql": a database of unit datasheets (stats such as toughness, wounds, points, \
weapons, abilities, keywords, factions)
- "both": the question needs the rulebook and the database

## Conversation so far
{history}

## Latest question
{question}

Reply with JSON only:
{{"route": "text" | "sql" | "both", "query": "<search query>"}}
The search query is the latest question made self-contained (resolve "it", "they", \
"that unit" from the conversation) and phrased with the game's own terms (for \
example Victory points, objective markers, Toughness, Command phase). If the \
question refers to something that is not in the conversation, keep it as written \
rather than guess a subject. Never add the words Warhammer or 40,000: every \
passage mentions them."""


def make_router_node(generate: Callable[..., str]):
    """Create the node that picks the retrieval channel(s) and rewrites the query.

    It also resets the per-turn fields, so contexts never leak between turns.
    """

    def router_node(state: AgentState) -> dict:
        prompt = PROMPT.format(history=history(state), question=question(state))
        route, query = "both", question(state)
        try:
            decision = json.loads(generate(prompt, json_output=True, temperature=0))
            if decision.get("route") in ROUTES:
                route = decision["route"]
            if isinstance(decision.get("query"), str) and decision["query"].strip():
                query = decision["query"]
        except (json.JSONDecodeError, AttributeError):
            logger.warning("router: unreadable decision, using route=both")
        logger.info("router: route=%s query=%r", route, query)
        return {"route": route, "query": query, "contexts": [], "sources": []}

    return router_node
