# 🧠 Context-Aware RAG Chatbot

> **AI/ML Engineering Internship — Task 4**  
> A production-grade, context-aware chatbot built with LangChain, Retrieval-Augmented Generation (RAG), ChromaDB, and Groq's Llama 3.1 70B.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-FF6B35)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1_70B-F55036)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [RAG Pipeline Explained](#-rag-pipeline-explained)
- [Evaluation](#-evaluation)
- [Technology Stack](#-technology-stack)
- [Internship Requirements Checklist](#-internship-requirements-checklist)

---

## 🎯 Overview

This project implements a **Context-Aware Conversational RAG Chatbot** that can answer questions about AI/ML research papers using:

- **Retrieval-Augmented Generation (RAG)** to ground answers in actual paper content
- **LangChain** for pipeline orchestration and memory management
- **ChromaDB** as a local, persistent vector store
- **HuggingFace sentence-transformers** for document and query embeddings
- **Groq (Llama 3.1 70B)** as the LLM backbone — blazing fast, free tier
- **Streamlit** for a beautiful, interactive dark-mode UI

The knowledge base is automatically populated by downloading **10 seminal AI/ML research papers** from arXiv (including "Attention Is All You Need", the original RAG paper, LLaMA papers, and more).

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  Chat UI · Source Viewer · Relevance Scores · Evaluation    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              LangChain ConversationalRetrievalChain          │
│                                                             │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  Question   │    │  Condense    │    │  Answer      │  │
│   │  Input      │───►│  Question    │───►│  Generation  │  │
│   └─────────────┘    │  (LLM)       │    │  (Groq LLM)  │  │
│                      └──────┬───────┘    └──────────────┘  │
│                             │ Standalone Q                   │
│                      ┌──────▼──────────────────────────┐   │
│                      │  MMR Retriever (ChromaDB)        │   │
│                      │  • fetch_k=20 candidates         │   │
│                      │  • select k=5 diverse results    │   │
│                      └──────┬──────────────────────────┘   │
│                             │                               │
│   ┌─────────────────────────▼──────────────────────────┐   │
│   │           ChromaDB Vector Store                    │   │
│   │   HuggingFace all-MiniLM-L6-v2 embeddings         │   │
│   │   384-dim · cosine similarity · persistent         │   │
│   └─────────────────────────┬──────────────────────────┘   │
│                             │                               │
│   ┌─────────────────────────▼──────────────────────────┐   │
│   │         Document Ingestion Pipeline                │   │
│   │   arXiv PDF → PyPDF → RecursiveTextSplitter       │   │
│   │   chunk_size=1000 · overlap=200                   │   │
│   └────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐ │
│   │  ConversationBufferWindowMemory (K=5 turns)          │ │
│   │  • Retains last 5 Q&A turns in LLM context          │ │
│   │  • Full history persisted separately for display    │ │
│   └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **RAG Pipeline** | Two-stage: condense question → retrieve → generate |
| **Memory** | `ConversationBufferWindowMemory` — retains last 5 turns |
| **Context Retention** | Multi-turn conversations with automatic question condensing |
| **Vector DB** | ChromaDB with persistent local storage |
| **Embeddings** | `all-MiniLM-L6-v2` — 384-dim, normalized, batched |
| **Retrieval** | MMR (Max Marginal Relevance) for diverse results |
| **Relevance Scores** | Per-source cosine similarity scores displayed in UI |
| **Source Viewer** | Shows which paper chunks were retrieved with previews |
| **Evaluation Suite** | 8-question benchmark with latency, relevance, diversity metrics |
| **Multi-model** | Switch between 5 Groq models from the UI |
| **Chat Export** | Download full conversation history as text |
| **Dark Mode UI** | Premium glassmorphism design with animations |
| **Caching** | PDFs cached locally — only downloads once |

---

## 📁 Project Structure

```
TASK_4Context-Aware Chatbot Using LangChain or RAG/
│
├── app.py                          # Streamlit application (entry point)
│
├── rag/                            # Core RAG pipeline
│   ├── __init__.py                 # Package exports
│   ├── document_loader.py          # arXiv PDF downloader + chunker
│   ├── embeddings.py               # HuggingFace embedding engine (singleton)
│   ├── vector_store.py             # ChromaDB management
│   ├── retriever.py                # MMR contextual retriever
│   ├── memory.py                   # Conversation memory manager
│   ├── chain.py                    # LangChain RAG chain assembly
│   └── evaluator.py                # Retrieval quality evaluation
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Centralized configuration
│
├── data/
│   ├── papers/                     # Downloaded arXiv PDFs (auto-created)
│   └── chroma_db/                  # Persistent vector store (auto-created)
│
├── tests/
│   ├── __init__.py
│   └── test_rag.py                 # 30+ unit tests across all modules
│
├── .env.example                    # Environment variable template
├── .env                            # Your API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Get a Free Groq API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up (free, takes 1 minute)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 2 — Clone & Setup

```bash
# Navigate to project
cd "TASK_4Context-Aware Chatbot Using LangChain or RAG"

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Configure API Key

```bash
# Copy the example file
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux

# Open .env and paste your Groq key:
# GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Step 5 — Build Knowledge Base

In the Streamlit sidebar:
1. Paste your Groq API key (or it loads from `.env` automatically)
2. Click **"🚀 Build Knowledge Base"**
3. Wait ~2-3 minutes while papers download and are indexed
4. **Start chatting!**

---

## ⚙️ Configuration

All settings are in [`config/settings.py`](config/settings.py). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `llama-3.1-70b-versatile` | Groq model |
| `LLM_TEMPERATURE` | `0.2` | Response creativity (0=deterministic) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVER_K` | `5` | Documents retrieved per query |
| `RETRIEVER_FETCH_K` | `20` | MMR candidate pool size |
| `RETRIEVER_LAMBDA_MULT` | `0.7` | MMR relevance vs diversity (1=max relevance) |
| `MEMORY_WINDOW_K` | `5` | Conversation turns retained in context |

Override any setting via environment variables in `.env`.

---

## 📖 Usage Guide

### Basic Chat

Type any AI/ML question in the chat box:
- *"What is the transformer architecture?"*
- *"How does RAG improve LLM accuracy?"*
- *"Explain chain-of-thought prompting."*

### Multi-Turn Conversation

The chatbot remembers context across turns:
```
You:  What is self-attention?
Bot:  Self-attention allows each token to attend to all other tokens...

You:  How is that different from cross-attention?
Bot:  [Correctly understands "that" refers to self-attention from above]
```

### Source Documents

After each answer, scroll down to see:
- Which paper each chunk came from
- The relevance score (cosine similarity %)
- A text preview of the retrieved content
- Direct arXiv links

### Evaluation Suite

Click **"📊 Run Evaluation Suite"** in the sidebar to run the 8-question benchmark and see:
- Average latency
- Average relevance scores
- Source diversity metrics
- Per-question detailed results

---

## 🔬 RAG Pipeline Explained

### 1. Document Ingestion
```
arXiv PDF → PyPDFLoader → RecursiveCharacterTextSplitter
  chunk_size=1000, overlap=200
  → List[Document] with metadata
```

### 2. Embedding
```
Document text → all-MiniLM-L6-v2 → 384-dim vector
  (normalized for cosine similarity)
```

### 3. Indexing
```
Vectors + metadata → ChromaDB (local persistent store)
  Batched in groups of 100 to avoid memory spikes
```

### 4. Retrieval (MMR)
```
Query → embed → ChromaDB similarity search (fetch_k=20)
  → MMR re-ranking (diversity + relevance)
  → Top k=5 diverse, relevant chunks
```

### 5. Question Condensing
```
Follow-up question + chat history 
  → LLM condenses to standalone question
  (enables correct retrieval for context-dependent queries)
```

### 6. Answer Generation
```
Standalone question + retrieved context + system prompt
  → Groq Llama 3.1 70B
  → Grounded, cited answer
```

### 7. Memory Update
```
(question, answer) → ConversationBufferWindowMemory
  → Included in next query's chat_history
```

---

## 📊 Evaluation

Run the evaluation suite from the Streamlit UI or directly:

```python
from rag.evaluator import RAGEvaluator
from rag.chain import RAGChain
from rag.retriever import ContextualRetriever
from rag.vector_store import VectorStoreManager

vsm = VectorStoreManager()
vsm.load()

chain = RAGChain(vsm)
retriever = ContextualRetriever(vsm)
evaluator = RAGEvaluator(chain, retriever)

results = evaluator.run_evaluation_suite()
print(evaluator.generate_report(results))
```

### Sample Benchmark Questions

1. What is the transformer architecture and how does attention work?
2. How does Retrieval-Augmented Generation (RAG) improve LLM accuracy?
3. What are the key differences between LLaMA and LLaMA 2?
4. Explain chain-of-thought prompting and its benefits.
5. What is RLHF and how is it used to align language models?
6. How does ReAct combine reasoning and acting in LLMs?
7. What is self-refinement in language model output?
8. What are the main challenges in factual accuracy for LLMs?

### Running Tests

```bash
# All tests
pytest tests/test_rag.py -v

# With coverage report
pytest tests/test_rag.py -v --cov=rag --cov-report=term-missing

# Specific module
pytest tests/test_rag.py::TestConversationMemoryManager -v
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | License |
|-----------|------------|---------|---------|
| **Framework** | LangChain | 0.3.x | MIT |
| **LLM** | Groq (Llama 3.1 70B) | - | Free tier |
| **Vector DB** | ChromaDB | 0.6.x | Apache 2.0 |
| **Embeddings** | sentence-transformers | 3.x | Apache 2.0 |
| **Embedding Model** | all-MiniLM-L6-v2 | - | Apache 2.0 |
| **UI** | Streamlit | 1.45.x | Apache 2.0 |
| **PDF Parsing** | pypdf | 5.x | MIT |
| **Testing** | pytest | 8.x | MIT |
| **Deep Learning** | PyTorch | 2.6.x | BSD |

**Total cost: $0** — All components are free and open source.

---

## ✅ Internship Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Use LangChain or RAG | ✅ | Both: `ConversationalRetrievalChain` + full RAG pipeline |
| Conversational Memory | ✅ | `ConversationBufferWindowMemory` (K=5 turns) |
| Context Retention | ✅ | Question condensing with chat history |
| Vector Database | ✅ | ChromaDB (local persistent) |
| Document Embeddings | ✅ | `all-MiniLM-L6-v2` via sentence-transformers |
| Streamlit Application | ✅ | Full dark-mode UI with chat + sources + evaluation |
| Multi-turn Conversations | ✅ | Full conversation history + context-aware follow-ups |
| RAG Pipeline | ✅ | `rag/chain.py` — full 2-stage pipeline |
| Document Loader | ✅ | `rag/document_loader.py` — arXiv PDF downloader |
| Embedding System | ✅ | `rag/embeddings.py` — singleton engine |
| Retriever | ✅ | `rag/retriever.py` — MMR with scored results |
| Memory Module | ✅ | `rag/memory.py` — window + persistence |
| LangChain Integration | ✅ | Full chain assembly in `rag/chain.py` |
| requirements.txt | ✅ | Pinned production dependencies |
| README.md | ✅ | This file |
| Evaluation Examples | ✅ | `rag/evaluator.py` + 8-question benchmark suite |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM application framework
- [Groq](https://console.groq.com) — Ultra-fast LLM inference
- [ChromaDB](https://trychroma.com) — Open-source vector database
- [Sentence Transformers](https://sbert.net) — State-of-the-art embeddings
- [arXiv](https://arxiv.org) — Open access research papers
