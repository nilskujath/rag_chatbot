from langchain_core.messages import AIMessage, HumanMessage

from rag_chatbot.agent.state import history, question


def conversation(turns: int) -> dict:
    messages = []
    for i in range(turns):
        messages += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]
    return {"messages": messages + [HumanMessage("latest")]}


def test_question_is_the_latest_user_message():
    assert question(conversation(2)) == "latest"


def test_history_excludes_the_latest_question_and_labels_roles():
    assert history(conversation(1)) == "User: q0\nAssistant: a0"


def test_history_keeps_only_the_last_turns(monkeypatch):
    monkeypatch.setattr("rag_chatbot.agent.state.settings.history_turns", 2)
    text = history(conversation(5))
    assert text.startswith("User: q3")
    assert text.endswith("Assistant: a4")
    assert "q2" not in text


def test_history_at_the_start_of_a_conversation():
    assert history(conversation(0)) == "(start of conversation)"
