"""Chroma persistent vector store with the same interface as PgVectorStore.

Used because Docker (needed for the pgvector starter) is not available on the
machine. Data is persisted on disk; one collection per strategy index; cosine
distance; Chroma's HNSW index, i.e. approximate nearest-neighbour search.
"""
from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

BATCH = 1000


class ChromaStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.path = Path(path)
        self.dimensions = dimensions
        self._client = chromadb.PersistentClient(path=str(self.path))

    def describe(self) -> dict[str, Any]:
        return {"store": "Chroma PersistentClient", "chromadb": chromadb.__version__,
                "path": f"starter/{self.path.name}", "collections": "day6_<strategy index>",
                "metric": "cosine", "search": "approximate (HNSW, Chroma defaults)"}

    def create_schema(self) -> None:
        """Collections are created on first use; nothing to do up front."""

    def _collection(self, strategy: str):
        return self._client.get_or_create_collection(
            name=f"day6_{strategy}",
            metadata={"hnsw:space": "cosine", "dimensions": self.dimensions},
        )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert by stable chunk ID, then delete this strategy's obsolete chunks
        for the documents being (re-)ingested."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in batch")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"vector has {len(vector)} dimensions, expected {self.dimensions}")
        collection = self._collection(strategy)
        metadatas = [
            {**c.metadata, "document_id": c.document_id, "source": c.source, "page": c.page}
            for c in chunks
        ]
        for start in range(0, len(chunks), BATCH):
            end = start + BATCH
            collection.upsert(ids=ids[start:end], embeddings=vectors[start:end],
                              documents=[c.text for c in chunks[start:end]],
                              metadatas=metadatas[start:end])
        keep = set(ids)
        for document_id in sorted({c.document_id for c in chunks}):
            stored = collection.get(where={"document_id": document_id}, include=[])["ids"]
            obsolete = [chunk_id for chunk_id in stored if chunk_id not in keep]
            for start in range(0, len(obsolete), BATCH):
                collection.delete(ids=obsolete[start:start + BATCH])

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Metadata equality filter applied inside Chroma, ranked by cosine distance."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query has {len(query_vector)} dimensions, expected {self.dimensions}")
        conditions = [{key: value} for key, value in (filters or {}).items()]
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = self._collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(result["ids"][0], result["documents"][0], result["metadatas"][0],
                    result["distances"][0]), start=1):
            metadata = dict(metadata)
            hits.append({
                "rank": rank, "chunk_id": chunk_id, "document_id": metadata.pop("document_id"),
                "source": metadata.pop("source"), "page": metadata.pop("page"),
                "text": text, "metadata": metadata, "score": 1 - float(distance),
            })
        return hits