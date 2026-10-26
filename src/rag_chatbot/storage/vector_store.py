import sqlite3
import struct

import sqlite_vec

from rag_chatbot.chunker.base import Chunk


class SqliteVecStore:
    """Chunks and their vectors in one SQLite file, searched with sqlite-vec.

    `vec0` performs an exact (brute force) nearest-neighbour scan, which is
    more than fast enough for a few hundred chunks and needs no server.
    """

    def __init__(self, db_path: str, dimension: int):
        self.dimension = dimension
        # langgraph runs nodes in worker threads; one query at a time, sharing is safe
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                section TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(embedding float[{self.dimension}]);
            """
        )

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Drop whatever is stored and insert these chunks, so a re-index is clean."""
        self.conn.executescript("DROP TABLE chunks; DROP TABLE vec_chunks;")
        self._create_tables()
        for row_id, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
            self.conn.execute(
                "INSERT INTO chunks (id, content, section) VALUES (?, ?, ?)",
                (row_id, chunk.content, chunk.section),
            )
            self.conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (row_id, _serialize(embedding)),
            )
        self.conn.commit()

    def search(self, query_embedding: list[float], k: int) -> list[Chunk]:
        """Return the k chunks closest to the query vector, nearest first."""
        rows = self.conn.execute(
            """SELECT c.content, c.section
               FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (_serialize(query_embedding), k),
        ).fetchall()
        return [Chunk(content, section) for content, section in rows]

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def _serialize(vector: list[float]) -> bytes:
    """Pack a vector into the little-endian float32 blob sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)
