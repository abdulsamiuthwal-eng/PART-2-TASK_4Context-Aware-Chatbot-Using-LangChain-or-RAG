"""
Retriever Module
================
Wraps the ChromaDB vector store in a LangChain retriever that uses
Maximum Marginal Relevance (MMR) to balance relevance with diversity.

MMR prevents retrieving 5 nearly-identical chunks from the same paragraph
by penalising redundancy — crucial for multi-paper knowledge bases.

Also provides a scored retriever that returns relevance scores
for display in the Streamlit UI.
"""

import logging
from typing import List, Tuple

from langchain.schema import Document, BaseRetriever
from langchain_chroma import Chroma

from rag.vector_store import VectorStoreManager
from config.settings import (
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ContextualRetriever:
    """
    Builds a LangChain-compatible retriever on top of ChromaDB.

    Retrieval Strategy: Maximum Marginal Relevance (MMR)
    - Fetch FETCH_K candidate chunks from ChromaDB
    - Re-rank to select K diverse, relevant chunks
    - lambda_mult controls relevance vs diversity trade-off

    Usage:
        retriever = ContextualRetriever(vector_store_manager)
        docs = retriever.retrieve("What is attention?")
        langchain_retriever = retriever.as_langchain_retriever()
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        k: int = RETRIEVER_K,
        fetch_k: int = RETRIEVER_FETCH_K,
        lambda_mult: float = RETRIEVER_LAMBDA_MULT,
        search_type: str = RETRIEVER_SEARCH_TYPE,
    ):
        self.vsm = vector_store_manager
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.search_type = search_type
        self._store: Chroma = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the top-K most relevant diverse documents for a query.
        Uses MMR by default.
        """
        store = self._get_store()

        if self.search_type == "mmr":
            docs = store.max_marginal_relevance_search(
                query=query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )
        else:
            docs = store.similarity_search(query=query, k=self.k)

        logger.debug("Retrieved %d chunks for query: %s...", len(docs), query[:60])
        return docs

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their relevance scores.
        Returns list of (Document, score) tuples — scores in [0.0, 1.0].
        Higher score = more relevant to query.
        """
        store = self._get_store()
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=self.k,
        )

        # Filter by minimum score threshold
        filtered = [
            (doc, score) for doc, score in results
            if score >= RETRIEVER_SCORE_THRESHOLD
        ]

        if not filtered and results:
            # If all below threshold, still return the best one
            filtered = [results[0]]

        logger.debug(
            "Retrieved %d scored chunks (threshold=%.2f) for: %s...",
            len(filtered), RETRIEVER_SCORE_THRESHOLD, query[:60]
        )
        return filtered

    def as_langchain_retriever(self) -> BaseRetriever:
        """
        Return a LangChain BaseRetriever compatible object.
        This is what gets passed to ConversationalRetrievalChain.
        """
        store = self._get_store()

        if self.search_type == "mmr":
            return store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": self.fetch_k,
                    "lambda_mult": self.lambda_mult,
                },
            )
        elif self.search_type == "similarity_score_threshold":
            return store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.k,
                    "score_threshold": RETRIEVER_SCORE_THRESHOLD,
                },
            )
        else:
            return store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k},
            )

    def get_retriever_info(self) -> dict:
        """Return metadata about the retriever configuration."""
        return {
            "search_type": self.search_type,
            "k": self.k,
            "fetch_k": self.fetch_k,
            "lambda_mult": self.lambda_mult,
            "score_threshold": RETRIEVER_SCORE_THRESHOLD,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        """Lazy-load the vector store on first access."""
        if self._store is None:
            self._store = self.vsm.get_store()
        return self._store
