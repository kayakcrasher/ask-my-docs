"""Vector store — SQLite + numpy. No external vector DB needed.

Why this approach:
  - SQLite is everywhere, zero install, works in Termux.
  - Vectors are stored as raw float32 BLOBs (compact, fast to read).
  - Cosine similarity is a matrix multiply — numpy handles thousands
    of chunks in microseconds. A vector DB only pays off at 100k+.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


class Store:
    def __init__(self, path: str | Path = "amd.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add(self, source: str, chunks: list[str], embeddings: np.ndarray) -> int:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) disagree"
            )
        rows = [
            (source, i, text, embeddings[i].astype(np.float32).tobytes())
            for i, text in enumerate(chunks)
        ]
        self.conn.executemany(
            "INSERT INTO chunks (source, chunk_index, text, embedding) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def delete_source(self, source: str) -> int:
        cur = self.conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self.conn.commit()
        return cur.rowcount

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, source, chunk_index, text, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return []
        dim = query_embedding.shape[0]
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            v = np.frombuffer(r[4], dtype=np.float32)
            if v.shape[0] != dim:
                raise ValueError(
                    f"embedding dim mismatch: DB has {v.shape[0]}, query has {dim}. "
                    "Re-index after changing embedding models."
                )
            mat[i] = v
        q = query_embedding.astype(np.float32)
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        q_n = q / (np.linalg.norm(q) + 1e-8)
        sims = mat_n @ q_n
        top = np.argsort(-sims)[: top_k]
        return [
            {
                "id": int(rows[i][0]),
                "source": rows[i][1],
                "chunk_index": int(rows[i][2]),
                "text": rows[i][3],
                "score": float(sims[i]),
            }
            for i in top
        ]

    def sources(self) -> list[tuple[str, int]]:
        return self.conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()

    def stats(self) -> dict[str, int]:
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sources = self.conn.execute(
            "SELECT COUNT(DISTINCT source) FROM chunks"
        ).fetchone()[0]
        return {"chunks": int(chunks), "sources": int(sources)}

    def close(self) -> None:
        self.conn.close()
