"""
RAG Chain Module
================
Assembles the complete LangChain Conversational RAG pipeline:

  Query → Condense with history → Retrieve from ChromaDB → Generate answer

Chain Architecture:
  ConversationalRetrievalChain
    ├── LLM:       ChatGroq (configurable model, default: llama-3.1-8b-instant)
    ├── Retriever: ContextualRetriever (MMR, ChromaDB)
    ├── Memory:    ConversationBufferWindowMemory (K=5 turns)
    └── Prompt:    Custom system prompt with context injection

Two-step chain:
  1. Condense question: rephrase follow-up questions using chat history
  2. Answer question: combine condensed question + retrieved docs → answer
"""

import os
import logging
from typing import Dict, Any, Optional, List

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_groq import ChatGroq

from rag.retriever import ContextualRetriever
from rag.memory import ConversationMemoryManager
from rag.vector_store import VectorStoreManager
from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that captures all necessary context.

If the follow-up is already standalone (no reference to prior turns), return it unchanged.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


QA_SYSTEM_TEMPLATE = """You are an expert AI/ML research assistant. Your knowledge base consists of 
seminal research papers on transformers, RAG, LLMs, and related topics.

Use ONLY the provided context to answer questions. Be precise, technical, and educational.
If the context does not contain enough information, clearly say so instead of guessing.

When answering:
- Cite specific papers when relevant (e.g., "According to 'Attention Is All You Need'...")
- Use technical terminology appropriately
- Structure complex answers with clear explanations
- If asked a follow-up, maintain awareness of the prior conversation

Context from Research Papers:
{context}"""

QA_HUMAN_TEMPLATE = """{question}"""

QA_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QA_SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(QA_HUMAN_TEMPLATE),
])


class RAGChain:
    """
    Full conversational RAG chain combining:
    - Groq LLM (model configurable via settings, default: llama-3.1-8b-instant)
    - ChromaDB retriever (MMR)
    - Window conversation memory
    - Source document tracking

    Usage:
        chain = RAGChain(vector_store_manager)
        result = chain.ask("What is the transformer architecture?")
        print(result["answer"])
        print(result["source_documents"])
        print(result["scores"])
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        memory_manager: Optional[ConversationMemoryManager] = None,
        model_name: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file or enter it in the sidebar.\n"
                "Get a free key at https://console.groq.com"
            )

        self.vsm = vector_store_manager
        self.memory_manager = memory_manager or ConversationMemoryManager()
        self.model_name = model_name

        # Build components
        self._llm = self._build_llm(model_name, temperature, max_tokens, api_key)
        self._retriever_wrapper = ContextualRetriever(vector_store_manager)
        self._chain = self._build_chain()

        logger.info("RAGChain initialized with model: %s", model_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main query interface. Process a question through the full RAG pipeline.

        Args:
            question: The user's natural language question.

        Returns:
            dict with keys:
                - answer (str): LLM-generated response
                - source_documents (List[Document]): Retrieved chunks
                - scores (List[float]): Relevance scores per source
                - question (str): The (possibly condensed) question
                - model (str): LLM model used
        """
        if not question.strip():
            return {
                "answer": "Please enter a question.",
                "source_documents": [],
                "scores": [],
                "question": question,
                "model": self.model_name,
            }

        try:
            # Get scored docs separately for UI display
            scored_docs = self._retriever_wrapper.retrieve_with_scores(question)

            # Run the conversational chain
            result = self._chain.invoke({"question": question})

            # Extract answer and sources
            answer = result.get("answer", "I could not generate a response.")
            source_docs = result.get("source_documents", [])
            scores = [score for _, score in scored_docs[:len(source_docs)]]

            # Pad scores if needed
            while len(scores) < len(source_docs):
                scores.append(0.0)

            # Update memory with this exchange
            self.memory_manager.add_exchange(question, answer)

            return {
                "answer": answer,
                "source_documents": source_docs,
                "scores": scores,
                "question": question,
                "model": self.model_name,
            }

        except Exception as exc:
            logger.error("RAG chain error for question '%s': %s", question[:60], exc)
            error_msg = self._format_error(exc)
            return {
                "answer": error_msg,
                "source_documents": [],
                "scores": [],
                "question": question,
                "model": self.model_name,
                "error": str(exc),
            }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory_manager.clear()
        # Rebuild chain with fresh memory
        self._chain = self._build_chain()
        logger.info("Chain memory cleared and rebuilt.")

    def get_chain_info(self) -> Dict[str, Any]:
        """Return metadata about the assembled chain."""
        return {
            "llm_model": self.model_name,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "retriever": self._retriever_wrapper.get_retriever_info(),
            "memory": self.memory_manager.get_memory_stats(),
            "vector_store": self.vsm.get_stats(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_api_key() -> str:
        """Resolve GROQ_API_KEY from environment or settings (runtime, not import-time)."""
        from config import settings
        return os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY or ""

    def _build_llm(self, model_name: str, temperature: float, max_tokens: int,
                   api_key: str = None) -> ChatGroq:
        """Initialize the Groq LLM client."""
        api_key = api_key or self._get_api_key()
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
        )

    def _build_chain(self) -> ConversationalRetrievalChain:
        """Assemble the ConversationalRetrievalChain with all components."""
        return ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=self._retriever_wrapper.as_langchain_retriever(),
            memory=self.memory_manager.langchain_memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_CHAT_PROMPT},
            return_source_documents=True,
            verbose=False,
            output_key="answer",
        )

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Convert exceptions to user-friendly error messages."""
        exc_str = str(exc).lower()
        if "api_key" in exc_str or "authentication" in exc_str or "401" in exc_str:
            return (
                "❌ **Invalid Groq API Key.** "
                "Please check your `.env` file and ensure `GROQ_API_KEY` is set correctly.\n\n"
                "Get your free key at [console.groq.com](https://console.groq.com)"
            )
        elif "rate" in exc_str or "429" in exc_str:
            return (
                "⏳ **Rate limit reached.** "
                "Groq's free tier has rate limits. Please wait a moment and try again."
            )
        elif "timeout" in exc_str:
            return "⌛ **Request timed out.** Please try again."
        elif "connection" in exc_str:
            return "🌐 **Connection error.** Please check your internet connection."
        else:
            return f"❌ **An error occurred:** {exc}\n\nPlease check the logs for details."
