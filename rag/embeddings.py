"""
Embedding Engine Module
=======================
Wraps HuggingFace sentence-transformers to produce dense vector embeddings
for documents and queries. Uses all-MiniLM-L6-v2 — 384-dim, fast, accurate.

Features:
- Lazy model loading (first-call initialization)
- Batch encoding for efficiency
- Consistent interface expected by LangChain / ChromaDB
- Device auto-detection (CPU / CUDA)
"""

import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Singleton-style wrapper around HuggingFace sentence-transformers.
    Returns a LangChain-compatible embeddings object.
    """

    _instance: "EmbeddingEngine" = None
    _embeddings: HuggingFaceEmbeddings = None

    def __new__(cls, *args, **kwargs):
        """Enforce singleton — only one model loaded per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        # Guard against re-initialization in singleton pattern
        if self._embeddings is not None:
            return

        self.model_name = model_name
        self.device = device
        self._embeddings = None  # Loaded on first access

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Lazy-load the embedding model on first access.
        This avoids a slow startup when the model is not yet needed.
        """
        if self._embeddings is None:
            logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "normalize_embeddings": True,   # Cosine similarity via dot product
                    "batch_size": 32,
                },
            )
            logger.info("Embedding model loaded successfully.")
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a dense vector."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings into dense vectors (batched)."""
        return self.embeddings.embed_documents(texts)

    @property
    def model_info(self) -> dict:
        """Return metadata about the loaded embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": 384,   # all-MiniLM-L6-v2 output dimension
            "max_seq_length": 256,  # Token limit for this model
            "normalize": True,
        }


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Module-level convenience function.
    Returns the shared HuggingFaceEmbeddings instance.
    Used by ChromaDB and LangChain components.
    """
    engine = EmbeddingEngine()
    return engine.embeddings
