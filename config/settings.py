"""
Centralized configuration for the Context-Aware RAG Chatbot.
All tunable parameters live here — no magic numbers scattered in code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ─── LLM Configuration (Groq) ─────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ─── Embedding Configuration ──────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = "cpu"          # Change to "cuda" if GPU available

# ─── Vector Store Configuration ───────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "arxiv_ai_ml_papers"
CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)

# ─── Document Processing ──────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000          # Characters per chunk
CHUNK_OVERLAP: int = 200        # Overlap between adjacent chunks
CHUNK_SEPARATORS: list = ["\n\n", "\n", ". ", " ", ""]

# ─── Retriever Configuration ──────────────────────────────────────────────────
RETRIEVER_K: int = 5            # Number of documents to retrieve
RETRIEVER_FETCH_K: int = 20     # Candidate pool size for MMR
RETRIEVER_LAMBDA_MULT: float = 0.7   # MMR diversity parameter (0=max diversity, 1=max relevance)
RETRIEVER_SEARCH_TYPE: str = "mmr"  # "similarity" | "mmr" | "similarity_score_threshold"
RETRIEVER_SCORE_THRESHOLD: float = 0.3   # Minimum relevance score

# ─── Memory Configuration ─────────────────────────────────────────────────────
MEMORY_WINDOW_K: int = 5        # Number of recent conversation turns to retain
MEMORY_KEY: str = "chat_history"
MEMORY_HUMAN_PREFIX: str = "Human"
MEMORY_AI_PREFIX: str = "Assistant"

# ─── arXiv Document Source ────────────────────────────────────────────────────
# Seminal AI/ML papers to build the knowledge base
ARXIV_PAPER_IDS: list = [
    "1706.03762",   # Attention Is All You Need (Transformers)
    "2005.11401",   # RAG: Retrieval-Augmented Generation
    "2302.00083",   # LLaMA: Open and Efficient Foundation Language Models
    "2307.09288",   # Llama 2: Open Foundation and Fine-Tuned Chat Models
    "2203.02155",   # Chain-of-Thought Prompting
    "2301.13379",   # InstructGPT / RLHF
    "2309.01219",   # FActScoring: Fine-grained Hallucination Evaluation
    "2210.11610",   # ReAct: Synergizing Reasoning and Acting
    "2304.01373",   # Self-Refine: Iterative Refinement with Self-Feedback
    "2302.13971",   # LangChain: Building Applications with LLMs through Composability
]

# Additional paper metadata for display
PAPER_TITLES: dict = {
    "1706.03762": "Attention Is All You Need",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2302.00083": "LLaMA: Open and Efficient Foundation Language Models",
    "2307.09288": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    "2203.02155": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "2301.13379": "Training language models to follow instructions with human feedback",
    "2309.01219": "FActScoring: Fine-Grained Atomic Evaluation of Factual Precision",
    "2210.11610": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2304.01373": "Self-Refine: Iterative Refinement with Self-Feedback",
    "2302.13971": "LangChain: Building Applications with LLMs",
}

# ─── UI Configuration ─────────────────────────────────────────────────────────
APP_TITLE: str = "🧠 RAG Research Assistant"
APP_SUBTITLE: str = "Context-Aware Chatbot powered by LangChain + Groq + ChromaDB"
APP_ICON: str = "🧠"
MAX_SOURCES_DISPLAYED: int = 3      # Number of source chunks shown in UI

# ─── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are an expert AI research assistant with deep knowledge of machine learning, 
natural language processing, and artificial intelligence. You help researchers and students understand 
complex AI/ML concepts by synthesizing information from retrieved research papers.

Guidelines:
- Answer based on the retrieved context from research papers
- Cite specific papers or sections when relevant  
- If the context doesn't contain enough information, say so clearly
- Be precise, technical, and educational
- Maintain awareness of the full conversation history

Retrieved Context:
{context}

Conversation History:
{chat_history}

Current Question: {question}

Provide a comprehensive, accurate answer:"""

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_QUESTIONS: list = [
    "What is the transformer architecture and how does attention work?",
    "How does Retrieval-Augmented Generation (RAG) improve LLM accuracy?",
    "What are the key differences between LLaMA and LLaMA 2?",
    "Explain chain-of-thought prompting and its benefits.",
    "What is RLHF and how is it used to align language models?",
    "How does ReAct combine reasoning and acting in LLMs?",
    "What is self-refinement in language model output?",
    "What are the main challenges in factual accuracy for LLMs?",
]
