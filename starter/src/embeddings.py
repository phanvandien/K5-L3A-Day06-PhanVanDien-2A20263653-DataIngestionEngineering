"""Real local embedding model via Sentence Transformers (no API key, cost 0)."""
from typing import Protocol

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Sentence Transformers adapter. Any failure raises; there is no mock fallback.

    Vectors are L2-normalized. Queries and documents are encoded the same way,
    except for E5-family models, which require "query: " / "passage: " prefixes.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.batch_size = batch_size
        name = model_name.lower()
        if "embeddinggemma-2" in name:
            # Text-only load (270M params) and the prefixes from the model card:
            # prompts "SearchQuery" and "Document". Needs sentence-transformers>=6.1.
            # float32: the checkpoint defaults to bfloat16, which the model card
            # advises against on CPU.
            import torch
            self._model = SentenceTransformer(
                model_name, config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"dtype": torch.float32})
            self.query_prefix = "task: search result | query: "
            self.document_prefix = "title: none | text: "
        else:
            self._model = SentenceTransformer(model_name)
            is_e5 = "e5" in name
            self.query_prefix = "query: " if is_e5 else ""
            self.document_prefix = "passage: " if is_e5 else ""
        probe = self._model.encode(["dimension probe"], normalize_embeddings=True)
        self.dimensions = int(probe.shape[1])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        if len(vectors) != len(texts):
            raise RuntimeError("embedding model returned a different number of vectors")
        return [[float(x) for x in vector] for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, question: str) -> list[float]:
        return self._encode([self.query_prefix + question])[0]