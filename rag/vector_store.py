"""
Vector Store Module
===================
Manages the ChromaDB persistent vector store lifecycle:
- Create / load existing collection
- Add document chunks with embeddings
- Reset / delete collection
- Expose collection stats

ChromaDB is stored locally in data/chroma_db — no cloud account required.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from langchain_chroma import Chroma
from langchain.schema import Document

from rag.embeddings import get_embeddings
from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages creation, loading, and population of a persistent ChromaDB store.
    
    Usage:
        manager = VectorStoreManager()
        manager.build(documents)        # Index documents
        store = manager.get_store()     # Get LangChain Chroma instance
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_dir: str = CHROMA_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_dir = str(persist_dir)
        self._store: Optional[Chroma] = None
        self._embeddings = get_embeddings()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, documents: List[Document], batch_size: int = 100) -> "VectorStoreManager":
        """
        Embed and index a list of Document chunks into ChromaDB.
        Uses batching to avoid memory spikes on large corpora.

        Args:
            documents: List of LangChain Document objects (already chunked).
            batch_size: Number of documents to embed per batch.

        Returns:
            self (for method chaining)
        """
        if not documents:
            raise ValueError("Cannot build vector store from empty document list.")

        logger.info("Building vector store with %d chunks...", len(documents))

        # Process in batches
        all_texts = [doc.page_content for doc in documents]
        all_metadatas = [doc.metadata for doc in documents]
        all_ids = [self._make_id(doc, i) for i, doc in enumerate(documents)]

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
        )

        # Add in batches
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_texts = all_texts[start:end]
            batch_metadatas = all_metadatas[start:end]
            batch_ids = all_ids[start:end]

            self._store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            logger.info("  Indexed batch %d–%d / %d", start, end, len(documents))

        logger.info("Vector store built successfully. Collection: %s", self.collection_name)
        return self

    def load(self) -> "VectorStoreManager":
        """
        Load an existing persistent ChromaDB collection from disk.
        Raises RuntimeError if the collection doesn't exist yet.
        """
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
        )

        count = self._store._collection.count()
        if count == 0:
            logger.warning("Loaded empty ChromaDB collection '%s'.", self.collection_name)
        else:
            logger.info("Loaded ChromaDB collection '%s' with %d vectors.", 
                       self.collection_name, count)
        return self

    def get_store(self) -> Chroma:
        """Return the underlying LangChain Chroma instance."""
        if self._store is None:
            self.load()
        return self._store

    def is_populated(self) -> bool:
        """Return True if the vector store exists on disk and has documents."""
        try:
            client = chromadb.PersistentClient(path=self.persist_dir)
            collections = client.list_collections()
            # chromadb 1.x returns list of strings; 0.x returned objects with .name
            existing = [
                c if isinstance(c, str) else c.name
                for c in collections
            ]
            if self.collection_name not in existing:
                return False
            collection = client.get_collection(self.collection_name)
            return collection.count() > 0
        except Exception as exc:
            logger.debug("is_populated check failed: %s", exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the current vector store."""
        if self._store is None:
            try:
                self.load()
            except Exception:
                return {"status": "not_initialized", "count": 0}

        try:
            count = self._store._collection.count()
            return {
                "status": "ready",
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "document_count": count,
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "count": 0}

    def reset(self) -> None:
        """Delete and recreate the collection — use with caution."""
        try:
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(self.collection_name)
            logger.info("Deleted collection '%s'.", self.collection_name)
        except ValueError:
            # Collection didn't exist — that's fine
            logger.info("Collection '%s' did not exist, nothing to delete.", self.collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection: %s", exc)
        self._store = None

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Direct similarity search returning (Document, score) pairs.
        Score is cosine similarity in [0, 1] — higher is more relevant.
        """
        store = self.get_store()
        return store.similarity_search_with_relevance_scores(query, k=k)

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(doc: Document, index: int) -> str:
        """
        Generate a deterministic unique ID for each document chunk.
        Based on paper_id + chunk_index to avoid duplicates on re-indexing.
        """
        paper_id = doc.metadata.get("paper_id", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", index)
        return f"{paper_id}__chunk_{chunk_idx:04d}"
