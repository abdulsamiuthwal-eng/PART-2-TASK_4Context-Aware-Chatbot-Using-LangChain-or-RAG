"""
Memory Module
=============
Manages conversational memory for multi-turn dialogue.

Strategy: ConversationBufferWindowMemory
- Retains last K human/AI message pairs (configurable, default K=5)
- Lightweight — no LLM calls needed for summarization
- Prevents unbounded token growth from accumulating history

Also provides:
- Serialization helpers (save/load history from JSON)
- Formatted history for display in Streamlit
- Clear / reset functionality
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from config.settings import (
    MEMORY_WINDOW_K,
    MEMORY_KEY,
    MEMORY_HUMAN_PREFIX,
    MEMORY_AI_PREFIX,
)

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Wraps LangChain's ConversationBufferWindowMemory with persistence,
    formatting, and Streamlit-friendly display helpers.

    Usage:
        memory = ConversationMemoryManager()
        memory.add_exchange("What is RAG?", "RAG stands for...")
        history = memory.get_formatted_history()
        memory.clear()
    """

    def __init__(
        self,
        window_k: int = MEMORY_WINDOW_K,
        memory_key: str = MEMORY_KEY,
        human_prefix: str = MEMORY_HUMAN_PREFIX,
        ai_prefix: str = MEMORY_AI_PREFIX,
    ):
        self.window_k = window_k
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

        self._memory = self._create_memory()
        self._full_history: List[Dict[str, str]] = []   # Full untruncated history

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def langchain_memory(self) -> ConversationBufferWindowMemory:
        """Return the LangChain memory object (pass directly to chains)."""
        return self._memory

    def add_exchange(self, human_message: str, ai_message: str) -> None:
        """
        Record a complete Q&A exchange in memory.
        Updates both the window buffer and the full history log.
        """
        # output key must match the output_key configured on the memory object
        self._memory.save_context(
            {"input": human_message},
            {"answer": ai_message},
        )
        self._full_history.append({
            "role": "human",
            "content": human_message,
            "timestamp": datetime.now().isoformat(),
        })
        self._full_history.append({
            "role": "assistant",
            "content": ai_message,
            "timestamp": datetime.now().isoformat(),
        })
        logger.debug("Memory updated. Window: %d messages.", self.get_turn_count() * 2)

    def get_formatted_history(self) -> str:
        """
        Return the current window memory as a formatted string.
        Used for injecting chat history into the RAG prompt.
        """
        messages: List[BaseMessage] = self._memory.chat_memory.messages
        if not messages:
            return ""

        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"{self.human_prefix}: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"{self.ai_prefix}: {msg.content}")
        return "\n".join(lines)

    def get_display_history(self) -> List[Dict[str, str]]:
        """
        Return conversation history formatted for Streamlit chat display.
        Returns list of {"role": "human"|"assistant", "content": "..."} dicts.
        """
        return [
            {"role": entry["role"], "content": entry["content"]}
            for entry in self._full_history
        ]

    def get_turn_count(self) -> int:
        """Return the number of complete Q&A turns in the window."""
        total_in_memory = len(self._memory.chat_memory.messages) // 2
        return min(total_in_memory, self.window_k)

    def get_total_turns(self) -> int:
        """Return total turns in full history (not just window)."""
        return len(self._full_history) // 2

    def clear(self) -> None:
        """Reset window memory and full history."""
        self._memory.clear()
        self._full_history.clear()
        logger.info("Conversation memory cleared.")

    def save_to_file(self, file_path: str) -> None:
        """Persist full conversation history to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._full_history, f, indent=2, ensure_ascii=False)
        logger.info("Conversation saved to %s", path)

    def load_from_file(self, file_path: str) -> None:
        """Load conversation history from a JSON file and rebuild memory."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)

        self.clear()
        self._full_history = history

        # Replay into LangChain memory window (last K turns only)
        pairs = []
        temp_human = None
        for entry in history:
            if entry["role"] == "human":
                temp_human = entry["content"]
            elif entry["role"] == "assistant" and temp_human:
                pairs.append((temp_human, entry["content"]))
                temp_human = None

        # Only load the last window_k pairs
        for human_msg, ai_msg in pairs[-self.window_k:]:
            self._memory.save_context(
                {"input": human_msg},
                {"answer": ai_msg},
            )
        logger.info("Loaded %d conversation turns from %s", len(pairs), path)

    def export_as_text(self) -> str:
        """Export full conversation as plain text for download."""
        lines = [f"# Conversation Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for entry in self._full_history:
            role = "You" if entry["role"] == "human" else "Assistant"
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            lines.append(f"[{ts}] {role}:\n{entry['content']}\n")
        return "\n".join(lines)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics for display."""
        return {
            "window_k": self.window_k,
            "current_window_turns": self.get_turn_count(),
            "total_turns": self.get_total_turns(),
            "window_messages": len(self._memory.chat_memory.messages),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _create_memory(self) -> ConversationBufferWindowMemory:
        """Instantiate a fresh LangChain window memory object."""
        return ConversationBufferWindowMemory(
            k=self.window_k,
            memory_key=self.memory_key,
            return_messages=True,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
            output_key="answer",
        )
