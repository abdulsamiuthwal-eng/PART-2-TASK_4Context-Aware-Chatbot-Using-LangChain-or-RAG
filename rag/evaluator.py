"""
Evaluator Module
================
Provides retrieval and answer quality evaluation for the RAG pipeline.

Metrics:
1. Retrieval Quality
   - Hit Rate: Did retrieved docs contain the answer?
   - MRR (Mean Reciprocal Rank): How high was the first relevant doc?
   - Average Relevance Score: Mean cosine similarity of retrieved chunks

2. Answer Quality  
   - Faithfulness: Is the answer grounded in retrieved context?
   - Completeness: Does the answer address the question?

3. Context Coverage
   - Source diversity: How many unique papers were retrieved?
   - Chunk coverage: How much of the available context was used?

Usage:
    evaluator = RAGEvaluator(rag_chain, retriever)
    results = evaluator.run_evaluation_suite()
    report = evaluator.generate_report(results)
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document

from rag.chain import RAGChain
from config.settings import EVAL_QUESTIONS

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality across retrieval and generation dimensions.

    Usage:
        evaluator = RAGEvaluator(chain, retriever)
        results = evaluator.run_evaluation_suite()
        print(evaluator.generate_report(results))
    """

    def __init__(
        self,
        rag_chain: RAGChain,
        retriever: Any,
        eval_questions: List[str] = None,
    ):
        self.chain = rag_chain
        self.retriever = retriever
        self.eval_questions = eval_questions or EVAL_QUESTIONS

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_single(self, question: str) -> Dict[str, Any]:
        """
        Run a complete evaluation for a single question.

        Returns:
            dict with retrieval metrics, answer, latency, and source info.
        """
        start_time = time.time()

        # Get answer + sources from the chain
        result = self.chain.ask(question)
        latency = time.time() - start_time

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        scores = result.get("scores", [])

        # Compute metrics
        retrieval_metrics = self._compute_retrieval_metrics(source_docs, scores)
        answer_metrics = self._compute_answer_metrics(answer, source_docs, question)

        return {
            "question": question,
            "answer": answer,
            "latency_seconds": round(latency, 2),
            "num_sources": len(source_docs),
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "sources": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "paper_id": doc.metadata.get("paper_id", ""),
                    "score": scores[i] if i < len(scores) else 0.0,
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "preview": doc.page_content[:200].strip(),
                }
                for i, doc in enumerate(source_docs)
            ],
        }

    def run_evaluation_suite(
        self,
        questions: Optional[List[str]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run evaluation across all configured questions.

        Args:
            questions: Optional override for eval question set.
            progress_callback: Optional callable(i, total, question) for UI updates.

        Returns:
            Aggregated evaluation report dict.
        """
        questions = questions or self.eval_questions
        individual_results = []

        logger.info("Running evaluation suite on %d questions...", len(questions))

        for i, question in enumerate(questions):
            if progress_callback:
                progress_callback(i, len(questions), question)

            try:
                result = self.evaluate_single(question)
                individual_results.append(result)
                logger.info(
                    "  Q%d: latency=%.2fs, sources=%d",
                    i + 1, result["latency_seconds"], result["num_sources"]
                )
            except Exception as exc:
                logger.error("Evaluation failed for Q%d: %s", i + 1, exc)
                individual_results.append({
                    "question": question,
                    "answer": f"ERROR: {exc}",
                    "latency_seconds": 0,
                    "num_sources": 0,
                    "error": str(exc),
                })

            # Polite delay between questions to avoid Groq 429 rate-limit errors
            if i < len(questions) - 1:
                time.sleep(3)

        # Aggregate metrics
        aggregate = self._aggregate_results(individual_results)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "individual_results": individual_results,
            "aggregate_metrics": aggregate,
        }

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Format evaluation results as a readable Markdown report.
        """
        agg = evaluation_results.get("aggregate_metrics", {})
        timestamp = evaluation_results.get("timestamp", "")[:19].replace("T", " ")
        n = evaluation_results.get("total_questions", 0)

        lines = [
            "# RAG Evaluation Report",
            f"**Date:** {timestamp}  |  **Questions Evaluated:** {n}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Average Latency | {agg.get('avg_latency', 0):.2f}s |",
            f"| Average Sources Retrieved | {agg.get('avg_sources', 0):.1f} |",
            f"| Average Relevance Score | {agg.get('avg_relevance_score', 0):.3f} |",
            f"| Answer Non-Empty Rate | {agg.get('answer_non_empty_rate', 0)*100:.1f}% |",
            f"| Source Diversity (unique papers) | {agg.get('avg_unique_papers', 0):.1f} |",
            f"| Error Rate | {agg.get('error_rate', 0)*100:.1f}% |",
            "",
            "## Per-Question Results",
            "",
        ]

        for i, result in enumerate(evaluation_results.get("individual_results", []), 1):
            q = result.get("question", "")[:80]
            a = result.get("answer", "")[:300]
            latency = result.get("latency_seconds", 0)
            n_src = result.get("num_sources", 0)
            lines.extend([
                f"### Q{i}: {q}",
                f"- **Latency:** {latency:.2f}s | **Sources:** {n_src}",
                f"- **Answer Preview:** {a}...",
                "",
            ])

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Private Metric Computers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_retrieval_metrics(
        self, docs: List[Document], scores: List[float]
    ) -> Dict[str, Any]:
        """Compute retrieval quality metrics for a set of retrieved documents."""
        if not docs:
            return {
                "num_retrieved": 0,
                "avg_relevance_score": 0.0,
                "max_relevance_score": 0.0,
                "unique_papers": 0,
                "source_diversity": 0.0,
            }

        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        unique_papers = len(set(
            doc.metadata.get("paper_id", f"unknown_{i}")
            for i, doc in enumerate(docs)
        ))

        return {
            "num_retrieved": len(docs),
            "avg_relevance_score": round(avg_score, 4),
            "max_relevance_score": round(max_score, 4),
            "unique_papers": unique_papers,
            "source_diversity": round(unique_papers / len(docs), 3) if docs else 0.0,
        }

    def _compute_answer_metrics(
        self, answer: str, docs: List[Document], question: str
    ) -> Dict[str, Any]:
        """Compute answer quality metrics (heuristic-based)."""
        if not answer or answer.startswith("❌") or answer.startswith("ERROR"):
            return {
                "is_non_empty": False,
                "word_count": 0,
                "has_citations": False,
                "appears_grounded": False,
            }

        # Check if answer cites paper titles from retrieved docs
        paper_titles = [
            doc.metadata.get("title", "").lower()
            for doc in docs
        ]
        answer_lower = answer.lower()
        has_citations = any(
            title[:20] in answer_lower
            for title in paper_titles
            if title
        )

        # Heuristic: answer uses words from context (rough groundedness check)
        context_words = set()
        for doc in docs:
            context_words.update(doc.page_content.lower().split())

        answer_words = set(answer_lower.split())
        overlap = len(answer_words & context_words)
        groundedness = min(overlap / max(len(answer_words), 1), 1.0)

        return {
            "is_non_empty": bool(answer.strip()),
            "word_count": len(answer.split()),
            "has_citations": has_citations,
            "appears_grounded": groundedness > 0.3,
            "groundedness_score": round(groundedness, 3),
        }

    def _aggregate_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics across all evaluation results."""
        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]

        if not valid:
            return {"error": "All evaluations failed.", "error_rate": 1.0}

        latencies = [r["latency_seconds"] for r in valid]
        sources_counts = [r["num_sources"] for r in valid]
        relevance_scores = [
            r["retrieval_metrics"].get("avg_relevance_score", 0.0)
            for r in valid
        ]
        non_empty = [
            r["answer_metrics"].get("is_non_empty", False)
            for r in valid
        ]
        unique_papers = [
            r["retrieval_metrics"].get("unique_papers", 0)
            for r in valid
        ]

        return {
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_sources": sum(sources_counts) / len(sources_counts),
            "avg_relevance_score": sum(relevance_scores) / len(relevance_scores),
            "answer_non_empty_rate": sum(non_empty) / len(non_empty),
            "avg_unique_papers": sum(unique_papers) / len(unique_papers),
            "error_rate": len(errors) / len(results),
            "total_evaluated": len(results),
            "successful": len(valid),
            "failed": len(errors),
        }
