"""Store and search vectors using PostgreSQL + pgvector.

One table holds every strategy; rows are keyed by (strategy, chunk_id).
Search is exact (sequential scan, no ANN index) with cosine distance ``<=>``.
"""
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

TABLE = "chunks"
COLUMN_FILTERS = ("document_id", "source", "page")


def _vector_literal(vector: list[float]) -> str:
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("vector contains NaN or infinity")
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int,
                 allowed_filters: set[str] | None = None) -> None:
        """``allowed_filters``: metadata keys callers may filter on (None = any key)."""
        if not database_url:
            raise ValueError("DATABASE_URL is empty")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.database_url = database_url
        self.dimensions = dimensions
        self.allowed_filters = allowed_filters

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=5)

    def versions(self) -> dict[str, str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_setting('server_version'), "
                "(SELECT extversion FROM pg_extension WHERE extname = 'vector')"
            ).fetchone()
        return {"postgresql": row[0], "pgvector": row[1]}

    def describe(self) -> dict[str, Any]:
        return {"store": "PostgreSQL + pgvector", **self.versions(), "table": TABLE,
                "metric": "cosine (<=>)", "search": "exact, no ANN index"}

    def create_schema(self) -> None:
        """Create the table; fail if an existing table has another vector size."""
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy    text    NOT NULL,
                    chunk_id    text    NOT NULL,
                    document_id text    NOT NULL,
                    source      text    NOT NULL,
                    page        integer NOT NULL,
                    text        text    NOT NULL,
                    metadata    jsonb   NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding   vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )"""
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_strategy_doc ON {TABLE} (strategy, document_id)"
            )
            existing = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,)
            ).fetchone()[0]
        if existing != self.dimensions:
            raise RuntimeError(
                f"table {TABLE} stores vector({existing}) but the model has {self.dimensions} "
                "dimensions; use the same model or drop the table deliberately"
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks by stable ID, then drop this strategy's obsolete
        chunks for the documents being (re-)ingested. All in one transaction."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in batch")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"vector has {len(vector)} dimensions, expected {self.dimensions}")
        rows = [
            (strategy, c.chunk_id, c.document_id, c.source, c.page, c.text,
             Jsonb(c.metadata), _vector_literal(v))
            for c, v in zip(chunks, vectors)
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""INSERT INTO {TABLE}
                        (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                            page = EXCLUDED.page, text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                    rows,
                )
                cur.execute(
                    f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = ANY(%s) "
                    "AND NOT (chunk_id = ANY(%s))",
                    (strategy, sorted({c.document_id for c in chunks}), ids),
                )

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT count(*) FROM {TABLE} WHERE strategy = %s", (strategy,)
            ).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter in SQL, rank by cosine distance, return top-k.

        ``document_id``, ``source`` and ``page`` filter on columns; any other key
        filters on the JSONB metadata by exact, type-sensitive equality (2025 is
        not "2025"). Every value is passed as a query parameter.
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query has {len(query_vector)} dimensions, expected {self.dimensions}")
        where, params = ["strategy = %s"], [strategy]
        metadata_filter: dict[str, Any] = {}
        for key, value in (filters or {}).items():
            if key in COLUMN_FILTERS:
                where.append(f"{key} = %s")  # key comes from the fixed tuple above
                params.append(value)
            elif self.allowed_filters is None or key in self.allowed_filters:
                metadata_filter[key] = value
            else:
                raise ValueError(f"filter on '{key}' is not permitted")
        if metadata_filter:
            where.append("metadata @> %s")
            params.append(Jsonb(metadata_filter))
        literal = _vector_literal(query_vector)
        sql = (
            f"SELECT chunk_id, document_id, source, page, text, metadata, "
            f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, [literal, *params, literal, top_k]).fetchall()
        return [
            {"rank": rank, "chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
             "text": r[4], "metadata": r[5], "score": float(r[6])}
            for rank, r in enumerate(rows, start=1)
        ]