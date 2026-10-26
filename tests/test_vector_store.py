from rag_chatbot.chunker import Chunk
from rag_chatbot.storage import SqliteVecStore

CHUNKS = [Chunk("north", "Compass > N"), Chunk("east", "Compass > E")]
VECTORS = [[0.0, 1.0], [1.0, 0.0]]


def test_search_returns_nearest_chunk_first(tmp_path):
    store = SqliteVecStore(str(tmp_path / "v.db"), dimension=2)
    store.replace(CHUNKS, VECTORS)
    assert store.search([0.1, 0.9], k=2) == [CHUNKS[0], CHUNKS[1]]
    assert store.search([0.9, 0.1], k=1) == [CHUNKS[1]]


def test_replace_is_idempotent(tmp_path):
    store = SqliteVecStore(str(tmp_path / "v.db"), dimension=2)
    store.replace(CHUNKS, VECTORS)
    store.replace(CHUNKS, VECTORS)
    assert len(store) == 2


def test_store_persists_across_connections(tmp_path):
    path = str(tmp_path / "v.db")
    SqliteVecStore(path, dimension=2).replace(CHUNKS, VECTORS)
    assert SqliteVecStore(path, dimension=2).search([0.0, 1.0], k=1) == [CHUNKS[0]]
