"""
Document Loader Module
======================
Downloads AI/ML research papers from arXiv and loads them into
LangChain Document objects with rich metadata for the RAG pipeline.

Supports:
- Auto-download from arXiv by paper ID
- PDF parsing with PyPDFLoader
- Recursive character-level text splitting
- Duplicate detection (skip already-downloaded papers)
- Metadata enrichment (title, authors, paper_id, chunk_index)
"""

import os
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Optional

import requests
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import (
    PAPERS_DIR,
    ARXIV_PAPER_IDS,
    PAPER_TITLES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)

logger = logging.getLogger(__name__)


class ArxivDocumentLoader:
    """
    Downloads arXiv papers as PDFs, parses them, chunks them,
    and returns a list of LangChain Document objects ready for embedding.
    """

    ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"
    ARXIV_ABS_URL = "https://arxiv.org/abs/{paper_id}"
    REQUEST_DELAY = 2.0        # seconds between requests (be polite to arXiv)
    REQUEST_TIMEOUT = 60       # seconds

    def __init__(
        self,
        paper_ids: List[str] = None,
        papers_dir: Path = PAPERS_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.paper_ids = paper_ids or ARXIV_PAPER_IDS
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHUNK_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load_all(self, progress_callback=None) -> List[Document]:
        """
        Main entry point: downloads all configured papers, parses them,
        chunks them, and returns enriched Document objects.

        Args:
            progress_callback: Optional callable(paper_id, status, message)
                               for Streamlit progress updates.

        Returns:
            List[Document] — all chunks from all papers, with metadata.
        """
        all_documents: List[Document] = []

        for i, paper_id in enumerate(self.paper_ids):
            try:
                if progress_callback:
                    progress_callback(paper_id, "downloading", f"Fetching {paper_id}...")

                pdf_path = self._download_paper(paper_id)

                if progress_callback:
                    progress_callback(paper_id, "parsing", f"Parsing {paper_id}...")

                docs = self._load_and_chunk(pdf_path, paper_id)
                all_documents.extend(docs)

                if progress_callback:
                    progress_callback(
                        paper_id, "done",
                        f"✅ {PAPER_TITLES.get(paper_id, paper_id)} — {len(docs)} chunks"
                    )

                logger.info("Loaded paper %s → %d chunks", paper_id, len(docs))

            except Exception as exc:
                logger.error("Failed to load paper %s: %s", paper_id, exc)
                if progress_callback:
                    progress_callback(paper_id, "error", f"❌ {paper_id}: {exc}")
                continue

            # Polite delay between downloads
            if i < len(self.paper_ids) - 1:
                time.sleep(self.REQUEST_DELAY)

        logger.info("Total documents loaded: %d chunks from %d papers",
                    len(all_documents), len(self.paper_ids))
        return all_documents

    def load_single(self, paper_id: str) -> List[Document]:
        """Load a single paper by arXiv ID."""
        pdf_path = self._download_paper(paper_id)
        return self._load_and_chunk(pdf_path, paper_id)

    def load_from_file(self, file_path: str, source_name: str = "uploaded") -> List[Document]:
        """
        Load and chunk a user-uploaded PDF file.
        Used when the user uploads their own documents via Streamlit.
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        paper_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
        return self._load_and_chunk(pdf_path, paper_id, title=source_name)

    def get_already_downloaded(self) -> List[str]:
        """Return list of paper IDs that already have a PDF on disk."""
        downloaded = []
        for paper_id in self.paper_ids:
            pdf_path = self.papers_dir / f"{paper_id.replace('/', '_')}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                downloaded.append(paper_id)
        return downloaded

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _download_paper(self, paper_id: str) -> Path:
        """
        Download a paper PDF from arXiv if not already cached.
        Returns the local path to the PDF file.
        """
        safe_id = paper_id.replace("/", "_")
        pdf_path = self.papers_dir / f"{safe_id}.pdf"

        if pdf_path.exists():
            # Validate file size — anything under 10 KB is likely corrupt/incomplete
            if pdf_path.stat().st_size > 10_000:
                logger.info("Cache hit: %s", pdf_path)
                return pdf_path
            else:
                logger.warning("Corrupt/incomplete cached PDF detected, deleting: %s", pdf_path)
                pdf_path.unlink(missing_ok=True)

        url = self.ARXIV_PDF_URL.format(paper_id=paper_id)
        logger.info("Downloading: %s", url)

        headers = {
            "User-Agent": "RAGChatbot/1.0 (Educational Research Tool; contact@example.com)"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=self.REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Saved PDF: %s (%.1f KB)", pdf_path, pdf_path.stat().st_size / 1024)
        return pdf_path

    def _load_and_chunk(
        self,
        pdf_path: Path,
        paper_id: str,
        title: Optional[str] = None,
    ) -> List[Document]:
        """
        Parse a PDF using PyPDFLoader and split it into chunks.
        Enriches each chunk with metadata.
        If the PDF is corrupt, deletes it so it can be re-downloaded next time.
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:
            logger.warning("PDF parse failed for %s (%s) — deleting corrupt file", pdf_path, exc)
            pdf_path.unlink(missing_ok=True)
            raise

        if not pages:
            logger.warning("No pages extracted from %s", pdf_path)
            return []

        # Split all pages into chunks
        chunks = self.text_splitter.split_documents(pages)

        # Enrich metadata on every chunk
        paper_title = title or PAPER_TITLES.get(paper_id, paper_id)
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "paper_id": paper_id,
                "title": paper_title,
                "source": f"arXiv:{paper_id}",
                "arxiv_url": self.ARXIV_ABS_URL.format(paper_id=paper_id),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "file_path": str(pdf_path),
            })

        return chunks
