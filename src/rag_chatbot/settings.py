"""Application settings, read from the environment and the .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Models, limits and tuneable parameters. Data file paths are CLI defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Google AI Studio)
    google_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    judge_model: str = "gemini-3.1-flash-lite"
    llm_requests_per_minute: int = 10
    llm_max_attempts: int = 5

    # Agent
    history_turns: int = 6  # messages shown to the LLM: 3 question-answer pairs
    vector_k: int = 5
    rerank_candidates: int = 20  # 0 turns the reranker off
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunker (lengths in characters)
    chunk_headers_to_split: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    chunk_max_length: int = 800
    chunk_overlap: int = 200
    chunk_min_length: int = 50
    chunk_separators: list[str] = ["\n\n", "\n", " ", ""]

    # Storage
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db_path: str = "data/vectors.db"

    # SQL retriever
    sql_db_path: str = "data/tables/wahadb.sqlite"
    sql_dictionary_path: str = "data/tables/SQL_AGENT_DATA_DICTIONARY.md"
    sql_max_rows: int = 50
    sql_max_cell_length: int = 200


settings = Settings()
