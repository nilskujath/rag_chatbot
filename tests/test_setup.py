import pytest

from rag_chatbot.agent import SetupError, check_setup

SETTINGS = "rag_chatbot.agent.workflow.settings"


def test_all_present_passes(tmp_path, monkeypatch):
    db = tmp_path / "wahadb.sqlite"
    db.touch()
    vectors = tmp_path / "vectors.db"
    vectors.touch()
    monkeypatch.setattr(f"{SETTINGS}.google_api_key", "key")
    monkeypatch.setattr(f"{SETTINGS}.sql_db_path", str(db))
    check_setup(str(vectors))


def test_every_missing_piece_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{SETTINGS}.google_api_key", "")
    monkeypatch.setattr(f"{SETTINGS}.sql_db_path", str(tmp_path / "missing.sqlite"))
    with pytest.raises(SetupError) as error:
        check_setup(str(tmp_path / "missing.db"))
    message = str(error.value)
    assert "GOOGLE_API_KEY" in message
    assert "missing.sqlite" in message
    assert "make index" in message
