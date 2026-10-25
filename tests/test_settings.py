from rag_chatbot.settings import Settings


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_LENGTH", "123")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    settings = Settings(_env_file=None)
    assert settings.chunk_max_length == 123
    assert settings.llm_model == "gemini-test"
    assert settings.chunk_overlap == 200


def test_unknown_env_keys_are_ignored(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc\nVECTOR_K=7\n")
    assert Settings(_env_file=env).vector_k == 7
