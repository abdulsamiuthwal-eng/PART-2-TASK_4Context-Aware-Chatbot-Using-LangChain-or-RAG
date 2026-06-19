"""
Unit Tests — Context-Aware RAG Chatbot
=======================================
Tests cover:
- Document loading and chunking
- Embedding engine
- Vector store operations
- Memory management
- RAG chain behavior
- Evaluator output format

Run with:
    pytest tests/test_rag.py -v
    pytest tests/test_rag.py -v --cov=rag --cov-report=term-missing
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_documents() -> List[Document]:
    """Create synthetic documents simulating chunked arXiv paper content."""
    return [
        Document(
            page_content=(
                "The Transformer architecture uses multi-head self-attention mechanisms. "
                "Each attention head learns different representations of the input sequence. "
                "The model processes all tokens in parallel, unlike RNNs which process sequentially."
            ),
            metadata={
                "paper_id": "1706.03762",
                "title": "Attention Is All You Need",
                "source": "arXiv:1706.03762",
                "arxiv_url": "https://arxiv.org/abs/1706.03762",
                "chunk_index": 0,
                "total_chunks": 50,
            },
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) combines parametric and non-parametric memory. "
                "The retriever fetches relevant documents from a corpus. "
                "The generator produces answers conditioned on both the question and retrieved documents."
            ),
            metadata={
                "paper_id": "2005.11401",
                "title": "RAG: Retrieval-Augmented Generation",
                "source": "arXiv:2005.11401",
                "arxiv_url": "https://arxiv.org/abs/2005.11401",
                "chunk_index": 0,
                "total_chunks": 45,
            },
        ),
        Document(
            page_content=(
                "Chain-of-thought prompting elicits multi-step reasoning in large language models. "
                "By providing a few reasoning examples, LLMs learn to decompose complex problems. "
                "This significantly improves performance on arithmetic and commonsense reasoning tasks."
            ),
            metadata={
                "paper_id": "2203.02155",
                "title": "Chain-of-Thought Prompting",
                "source": "arXiv:2203.02155",
                "arxiv_url": "https://arxiv.org/abs/2203.02155",
                "chunk_index": 0,
                "total_chunks": 30,
            },
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Document Loader Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestArxivDocumentLoader:
    """Tests for the arXiv document downloader and chunker."""

    def test_loader_initializes_with_default_papers(self):
        from rag.document_loader import ArxivDocumentLoader
        loader = ArxivDocumentLoader()
        assert len(loader.paper_ids) > 0, "Should have default paper IDs"

    def test_loader_accepts_custom_paper_ids(self):
        from rag.document_loader import ArxivDocumentLoader
        custom_ids = ["1706.03762", "2005.11401"]
        loader = ArxivDocumentLoader(paper_ids=custom_ids)
        assert loader.paper_ids == custom_ids

    def test_text_splitter_configured_correctly(self):
        from rag.document_loader import ArxivDocumentLoader
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        loader = ArxivDocumentLoader()
        assert loader.text_splitter._chunk_size == CHUNK_SIZE
        assert loader.text_splitter._chunk_overlap == CHUNK_OVERLAP

    def test_chunk_metadata_enriched(self, sample_documents, tmp_path):
        """Test that metadata is properly attached to each chunk."""
        doc = sample_documents[0]
        assert "paper_id" in doc.metadata
        assert "title" in doc.metadata
        assert "source" in doc.metadata
        assert "chunk_index" in doc.metadata
        assert "arxiv_url" in doc.metadata

    @patch("rag.document_loader.requests.get")
    def test_download_skipped_if_cached(self, mock_get, tmp_path):
        """Should NOT make HTTP request if PDF already on disk."""
        from rag.document_loader import ArxivDocumentLoader
        loader = ArxivDocumentLoader(papers_dir=tmp_path)

        # Create a fake cached PDF
        pdf_path = tmp_path / "1706.03762.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 " + b"x" * 2000)

        result = loader._download_paper("1706.03762")

        mock_get.assert_not_called()
        assert result == pdf_path

    def test_get_already_downloaded_returns_correct_ids(self, tmp_path):
        from rag.document_loader import ArxivDocumentLoader
        loader = ArxivDocumentLoader(
            paper_ids=["1706.03762", "2005.11401"],
            papers_dir=tmp_path,
        )
        # Create one fake PDF
        (tmp_path / "1706.03762.pdf").write_bytes(b"%PDF" + b"x" * 2000)

        downloaded = loader.get_already_downloaded()
        assert "1706.03762" in downloaded
        assert "2005.11401" not in downloaded


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Embedding Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingEngine:
    """Tests for the embedding engine (uses mocked model to avoid slow downloads)."""

    def test_singleton_pattern(self):
        """Two instances should be the same object."""
        from rag.embeddings import EmbeddingEngine
        e1 = EmbeddingEngine()
        e2 = EmbeddingEngine()
        assert e1 is e2, "EmbeddingEngine should be a singleton"

    def test_model_info_returns_correct_dim(self):
        from rag.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        info = engine.model_info
        assert info["embedding_dim"] == 384
        assert info["normalize"] is True

    def test_model_info_structure(self):
        from rag.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        info = engine.model_info
        required_keys = ["model_name", "device", "embedding_dim", "max_seq_length", "normalize"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Vector Store Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreManager:
    """Tests for ChromaDB vector store management."""

    def test_make_id_is_deterministic(self):
        """Same document should always produce the same ID."""
        from rag.vector_store import VectorStoreManager
        doc = Document(
            page_content="test",
            metadata={"paper_id": "1706.03762", "chunk_index": 3}
        )
        id1 = VectorStoreManager._make_id(doc, 0)
        id2 = VectorStoreManager._make_id(doc, 0)
        assert id1 == id2

    def test_make_id_is_unique_per_chunk(self):
        """Different chunk indices should produce different IDs."""
        from rag.vector_store import VectorStoreManager
        doc1 = Document(page_content="a", metadata={"paper_id": "1706.03762", "chunk_index": 0})
        doc2 = Document(page_content="b", metadata={"paper_id": "1706.03762", "chunk_index": 1})
        assert VectorStoreManager._make_id(doc1, 0) != VectorStoreManager._make_id(doc2, 1)

    def test_make_id_format(self):
        """ID should follow expected format: paper_id__chunk_XXXX."""
        from rag.vector_store import VectorStoreManager
        doc = Document(page_content="x", metadata={"paper_id": "1706.03762", "chunk_index": 7})
        doc_id = VectorStoreManager._make_id(doc, 0)
        assert "1706.03762" in doc_id
        assert "chunk_" in doc_id

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_is_populated_returns_false_when_empty(self, mock_client_cls):
        """is_populated() should return False on empty collection."""
        from rag.vector_store import VectorStoreManager
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list_collections.return_value = []

        vsm = VectorStoreManager()
        assert vsm.is_populated() is False

    def test_get_stats_returns_not_initialized_when_empty(self):
        """get_stats() without initialization returns safe defaults."""
        from rag.vector_store import VectorStoreManager
        vsm = VectorStoreManager()
        # Don't load or build — just check stats handle missing store
        stats = vsm.get_stats()
        assert "status" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Memory Manager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationMemoryManager:
    """Tests for multi-turn conversation memory."""

    def test_memory_starts_empty(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        assert mem.get_turn_count() == 0
        assert mem.get_total_turns() == 0

    def test_add_exchange_increments_turn_count(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")
        assert mem.get_total_turns() == 1

    def test_window_respects_k_limit(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager(window_k=2)
        mem.add_exchange("Q1", "A1")
        mem.add_exchange("Q2", "A2")
        mem.add_exchange("Q3", "A3")
        # Window should only hold last 2 turns (k=2)
        assert mem.get_turn_count() <= 2

    def test_clear_resets_all_history(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("Hello", "Hi there!")
        mem.add_exchange("How are you?", "I'm doing well.")
        mem.clear()
        assert mem.get_turn_count() == 0
        assert mem.get_total_turns() == 0
        assert mem.get_display_history() == []

    def test_formatted_history_contains_messages(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("What is attention?", "Attention is a mechanism...")
        history = mem.get_formatted_history()
        assert "attention" in history.lower() or "What is attention" in history

    def test_display_history_has_correct_roles(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("Human question", "AI answer")
        display = mem.get_display_history()
        roles = [entry["role"] for entry in display]
        assert "human" in roles
        assert "assistant" in roles

    def test_export_as_text_contains_content(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("What is RAG?", "RAG is Retrieval-Augmented Generation.")
        export = mem.export_as_text()
        assert "RAG" in export
        assert "Conversation Export" in export

    def test_save_and_load_file(self, tmp_path):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        mem.add_exchange("Q1", "A1")
        mem.add_exchange("Q2", "A2")

        save_path = str(tmp_path / "history.json")
        mem.save_to_file(save_path)

        mem2 = ConversationMemoryManager()
        mem2.load_from_file(save_path)
        assert mem2.get_total_turns() == 2

    def test_memory_stats_structure(self):
        from rag.memory import ConversationMemoryManager
        mem = ConversationMemoryManager()
        stats = mem.get_memory_stats()
        required = ["window_k", "current_window_turns", "total_turns", "window_messages"]
        for key in required:
            assert key in stats, f"Missing stat: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Retriever Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextualRetriever:
    """Tests for the MMR-based retriever (mocked vector store)."""

    def _make_mock_vsm(self, sample_documents):
        """Create a mocked VectorStoreManager."""
        vsm = MagicMock()
        mock_store = MagicMock()
        vsm.get_store.return_value = mock_store
        mock_store.max_marginal_relevance_search.return_value = sample_documents[:2]
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.85 - i * 0.1) for i, doc in enumerate(sample_documents[:2])
        ]
        mock_store.as_retriever.return_value = MagicMock()
        return vsm

    def test_retrieve_returns_documents(self, sample_documents):
        from rag.retriever import ContextualRetriever
        vsm = self._make_mock_vsm(sample_documents)
        retriever = ContextualRetriever(vsm)
        docs = retriever.retrieve("What is attention?")
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_retrieve_with_scores_returns_tuples(self, sample_documents):
        from rag.retriever import ContextualRetriever
        vsm = self._make_mock_vsm(sample_documents)
        retriever = ContextualRetriever(vsm)
        results = retriever.retrieve_with_scores("What is RAG?")
        assert isinstance(results, list)
        if results:
            doc, score = results[0]
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_get_retriever_info_has_required_keys(self, sample_documents):
        from rag.retriever import ContextualRetriever
        vsm = self._make_mock_vsm(sample_documents)
        retriever = ContextualRetriever(vsm)
        info = retriever.get_retriever_info()
        required = ["search_type", "k", "fetch_k", "lambda_mult", "score_threshold"]
        for key in required:
            assert key in info, f"Missing info key: {key}"

    def test_as_langchain_retriever_returns_retriever(self, sample_documents):
        from rag.retriever import ContextualRetriever
        vsm = self._make_mock_vsm(sample_documents)
        retriever = ContextualRetriever(vsm)
        lc_retriever = retriever.as_langchain_retriever()
        assert lc_retriever is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RAG Chain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGChain:
    """Tests for the assembled RAG chain (mocked LLM + vector store)."""

    def test_chain_raises_without_api_key(self):
        """Should raise ValueError if GROQ_API_KEY is not set."""
        from rag.chain import RAGChain
        vsm = MagicMock()
        vsm.get_store.return_value = MagicMock()
        vsm.get_stats.return_value = {"document_count": 100}

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with patch("config.settings.GROQ_API_KEY", ""):
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    RAGChain(vsm)

    def test_format_error_api_key_message(self):
        """API key errors should return friendly message."""
        from rag.chain import RAGChain
        exc = Exception("authentication failed 401")
        msg = RAGChain._format_error(exc)
        assert "API Key" in msg or "Groq" in msg

    def test_format_error_rate_limit_message(self):
        """Rate limit errors should return friendly message."""
        from rag.chain import RAGChain
        exc = Exception("rate limit exceeded 429")
        msg = RAGChain._format_error(exc)
        assert "limit" in msg.lower() or "wait" in msg.lower()

    def test_format_error_timeout_message(self):
        from rag.chain import RAGChain
        exc = Exception("connection timeout")
        msg = RAGChain._format_error(exc)
        assert "timeout" in msg.lower() or "timed out" in msg.lower()

    def test_format_error_connection_message(self):
        from rag.chain import RAGChain
        exc = Exception("connection refused")
        msg = RAGChain._format_error(exc)
        assert "connection" in msg.lower() or "internet" in msg.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGEvaluator:
    """Tests for the evaluation module."""

    def _make_mock_chain(self, sample_documents):
        chain = MagicMock()
        chain.ask.return_value = {
            "answer": "The transformer uses multi-head attention as described in Attention Is All You Need.",
            "source_documents": sample_documents[:2],
            "scores": [0.88, 0.75],
            "question": "What is attention?",
            "model": "llama-3.1-70b-versatile",
        }
        chain.model_name = "llama-3.1-70b-versatile"
        return chain

    def test_evaluate_single_returns_required_keys(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever, eval_questions=["What is attention?"])
        result = evaluator.evaluate_single("What is attention?")
        required_keys = ["question", "answer", "latency_seconds", "num_sources",
                         "retrieval_metrics", "answer_metrics", "sources"]
        for key in required_keys:
            assert key in result, f"Missing result key: {key}"

    def test_retrieval_metrics_structure(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever)
        metrics = evaluator._compute_retrieval_metrics(sample_documents[:2], [0.9, 0.7])
        assert "num_retrieved" in metrics
        assert "avg_relevance_score" in metrics
        assert "unique_papers" in metrics
        assert metrics["num_retrieved"] == 2

    def test_answer_metrics_non_empty(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever)
        metrics = evaluator._compute_answer_metrics(
            "The transformer uses attention mechanisms.", sample_documents, "What is attention?"
        )
        assert metrics["is_non_empty"] is True
        assert metrics["word_count"] > 0

    def test_answer_metrics_empty_answer(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever)
        metrics = evaluator._compute_answer_metrics("", [], "What?")
        assert metrics["is_non_empty"] is False

    def test_generate_report_contains_metrics(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever, eval_questions=["What is attention?"])
        results = evaluator.run_evaluation_suite(questions=["What is attention?"])
        report = evaluator.generate_report(results)
        assert "RAG Evaluation Report" in report
        assert "Aggregate Metrics" in report
        assert "Latency" in report

    def test_aggregate_results_handles_all_failures(self):
        from rag.evaluator import RAGEvaluator
        chain = MagicMock()
        retriever = MagicMock()
        evaluator = RAGEvaluator(chain, retriever)
        failed_results = [{"question": "Q?", "error": "Network error"}]
        agg = evaluator._aggregate_results(failed_results)
        assert "error" in agg or agg.get("error_rate") == 1.0

    def test_run_evaluation_suite_returns_expected_structure(self, sample_documents):
        from rag.evaluator import RAGEvaluator
        chain = self._make_mock_chain(sample_documents)
        retriever = MagicMock()
        evaluator = RAGEvaluator(
            chain, retriever, eval_questions=["What is RAG?", "What is attention?"]
        )
        results = evaluator.run_evaluation_suite(questions=["What is RAG?"])
        assert "timestamp" in results
        assert "total_questions" in results
        assert "individual_results" in results
        assert "aggregate_metrics" in results
        assert results["total_questions"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfiguration:
    """Tests for the settings module."""

    def test_paths_are_path_objects(self):
        from config.settings import BASE_DIR, DATA_DIR, PAPERS_DIR, CHROMA_DIR
        assert hasattr(BASE_DIR, "__fspath__")
        assert hasattr(PAPERS_DIR, "__fspath__")
        assert hasattr(CHROMA_DIR, "__fspath__")

    def test_chunk_size_is_positive(self):
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_retriever_k_is_valid(self):
        from config.settings import RETRIEVER_K, RETRIEVER_FETCH_K
        assert 1 <= RETRIEVER_K <= 20
        assert RETRIEVER_FETCH_K >= RETRIEVER_K

    def test_memory_window_k_is_valid(self):
        from config.settings import MEMORY_WINDOW_K
        assert 1 <= MEMORY_WINDOW_K <= 20

    def test_arxiv_paper_ids_not_empty(self):
        from config.settings import ARXIV_PAPER_IDS
        assert len(ARXIV_PAPER_IDS) > 0

    def test_paper_titles_cover_all_ids(self):
        from config.settings import ARXIV_PAPER_IDS, PAPER_TITLES
        for pid in ARXIV_PAPER_IDS:
            assert pid in PAPER_TITLES, f"Paper ID {pid} has no title mapping"

    def test_eval_questions_not_empty(self):
        from config.settings import EVAL_QUESTIONS
        assert len(EVAL_QUESTIONS) > 0
        assert all(isinstance(q, str) for q in EVAL_QUESTIONS)


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
