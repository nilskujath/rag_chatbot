import json

from langchain_core.messages import HumanMessage

from rag_chatbot.agent import build_agent
from rag_chatbot.agent.nodes.router import make_router_node
from rag_chatbot.agent.workflow import after_router, after_rulebook

THREAD = {"configurable": {"thread_id": "test"}}


def ask(agent, question: str) -> dict:
    return agent.invoke({"messages": [("user", question)]}, THREAD)


def test_router_reads_route_and_query(fake_generate):
    state = make_router_node(fake_generate)({"messages": [HumanMessage("grenades?")]})
    assert state["route"] == "text"
    assert state["query"] == "rewritten query"
    assert "grenades?" in fake_generate.call_args.args[0]
    assert state["contexts"] == [] and state["sources"] == []


def test_router_falls_back_to_both_on_bad_json():
    node = make_router_node(lambda prompt, json_output, temperature: "not json")
    state = node({"messages": [HumanMessage("anything")]})
    assert state["route"] == "both"
    assert state["query"] == "anything"


def test_router_rejects_unknown_route():
    decision = json.dumps({"route": "pdf", "query": "q"})
    node = make_router_node(lambda prompt, json_output, temperature: decision)
    assert node({"messages": [HumanMessage("x")]})["route"] == "both"


def test_router_keeps_the_question_when_query_is_not_text():
    decision = json.dumps({"route": "sql", "query": ["a", "list"]})
    node = make_router_node(lambda prompt, json_output, temperature: decision)
    assert node({"messages": [HumanMessage("x")]})["query"] == "x"


def test_edges():
    assert after_router({"route": "sql"}) == "database"
    assert after_router({"route": "text"}) == "rulebook"
    assert after_router({"route": "both"}) == "rulebook"
    assert after_rulebook({"route": "both"}) == "database"
    assert after_rulebook({"route": "text"}) == "final_answer"


def test_text_route_uses_rulebook_only(fake_generate, fake_stream, rulebook, database):
    agent = build_agent(fake_generate, fake_stream, rulebook, database)
    state = ask(agent, "What is the rule for grenades?")

    assert state["sources"] == ["Rulebook"]
    assert rulebook.queries == ["rewritten query"]
    assert database.queries == []
    assert state["contexts"] == [
        "##### Rulebook, passage 1 #####\nsection: Shooting\nRoll to hit."
    ]
    assert state["messages"][-1].content == "The answer."


def test_both_route_numbers_passages_across_channels(fake_stream, rulebook, database):
    decision = json.dumps({"route": "both", "query": "q"})
    agent = build_agent(
        lambda p, json_output, temperature: decision, fake_stream, rulebook, database
    )
    state = ask(agent, "Toughness of a space marine and how to wound it?")

    assert state["sources"] == ["Rulebook", "Game database"]
    assert [c.split("\n")[0] for c in state["contexts"]] == [
        "##### Rulebook, passage 1 #####",
        "##### Game database, passage 2 #####",
    ]


def test_second_turn_resets_contexts_and_sees_history(
    fake_generate, fake_stream, rulebook, database
):
    agent = build_agent(fake_generate, fake_stream, rulebook, database)
    ask(agent, "first question")
    state = ask(agent, "follow-up question")

    assert len(state["contexts"]) == 1
    assert len(state["messages"]) == 4
    router_prompt = fake_generate.call_args_list[-1].args[0]
    answer_prompt = fake_stream.call_args_list[-1].args[0]
    assert "User: first question" in router_prompt
    assert "Assistant: The answer." in answer_prompt
