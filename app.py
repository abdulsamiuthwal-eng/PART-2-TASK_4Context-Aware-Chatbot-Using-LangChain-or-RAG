"""
Context-Aware RAG Chatbot — Streamlit Application
==================================================
Full-featured Streamlit UI with:
- Dark-mode sidebar with setup controls
- Multi-turn chat interface (st.chat_message style)
- Source document viewer with relevance scores
- Memory stats panel
- Knowledge base build / status
- Evaluation suite runner
- Chat export functionality
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ─── Path setup (so imports work from project root) ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    GROQ_API_KEY, LLM_MODEL, ARXIV_PAPER_IDS,
    PAPER_TITLES, MAX_SOURCES_DISPLAYED,
    EVAL_QUESTIONS,
)
from rag.document_loader import ArxivDocumentLoader
from rag.embeddings import EmbeddingEngine
from rag.vector_store import VectorStoreManager
from rag.retriever import ContextualRetriever
from rag.memory import ConversationMemoryManager
from rag.chain import RAGChain
from rag.evaluator import RAGEvaluator

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"**{APP_TITLE}** — Context-Aware RAG Chatbot powered by LangChain + Groq + ChromaDB",
        "Report a bug": "https://github.com",
    },
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global Dark Theme ───────────────────────────────────── */
:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #1a1d27;
    --bg-card:       #1e2130;
    --bg-input:      #252836;
    --accent-purple: #7c3aed;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --border:        #2d3148;
    --border-accent: #7c3aed40;
    --shadow:        0 4px 24px rgba(0,0,0,0.4);
}

/* ── App background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #0d1025 50%, #0f1117 100%);
    color: var(--text-primary);
}

/* ── Hide Streamlit branding ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Header Banner ──────────────────────────────────────── */
.rag-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-accent);
    box-shadow: var(--shadow), 0 0 60px rgba(124,58,237,0.15);
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.rag-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}
.rag-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Metric Cards ───────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: var(--accent-purple);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* ── Source Cards ───────────────────────────────────────── */
.source-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-purple);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
}
.source-title {
    font-weight: 600;
    color: var(--accent-blue);
    font-size: 0.88rem;
}
.source-score {
    font-size: 0.78rem;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-weight: 600;
}
.score-high   { background: #065f4620; color: #34d399; border: 1px solid #34d39940; }
.score-medium { background: #78350f20; color: #fbbf24; border: 1px solid #fbbf2440; }
.score-low    { background: #7f1d1d20; color: #f87171; border: 1px solid #f8717140; }

/* ── Chat messages ──────────────────────────────────────── */
.stChatMessage {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141e 0%, #0f1117 100%);
    border-right: 1px solid var(--border);
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
    border: 1px solid var(--border);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(124,58,237,0.3);
}

/* ── Progress / status ──────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-ready   { background: #065f4620; color: #34d399; }
.status-building{ background: #1e3a8a20; color: #60a5fa; }
.status-error   { background: #7f1d1d20; color: #f87171; }

/* ── Section headers ────────────────────────────────────── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.5rem 0 0.25rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}

/* ── Scrollable source panel ────────────────────────────── */
.source-panel {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 0.25rem;
}

/* ── Tag pills ──────────────────────────────────────────── */
.tag-pill {
    display: inline-block;
    background: var(--bg-input);
    color: var(--text-secondary);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 0.15rem;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        "messages": [],           # Chat history for display
        "rag_chain": None,        # RAGChain instance
        "memory_manager": None,   # ConversationMemoryManager instance
        "vsm": None,              # VectorStoreManager instance
        "kb_built": False,        # Knowledge base ready?
        "kb_doc_count": 0,        # Number of indexed chunks
        "kb_paper_count": 0,      # Number of papers indexed
        "last_sources": [],       # Sources from last query
        "last_scores": [],        # Scores from last query
        "api_key_set": bool(GROQ_API_KEY),
        "user_api_key": GROQ_API_KEY or "",
        "selected_model": LLM_MODEL,
        "eval_results": None,
        "show_eval": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the complete sidebar UI."""
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:3rem;">🧠</div>
            <div style="font-weight:800; font-size:1.1rem; 
                        background: linear-gradient(135deg,#a78bfa,#60a5fa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                RAG Research Assistant
            </div>
            <div style="color:#64748b; font-size:0.75rem;">LangChain · Groq · ChromaDB</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔑 API Configuration</div>', unsafe_allow_html=True)

        # API Key input
        api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.user_api_key,
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
            key="api_key_input",
        )
        if api_key != st.session_state.user_api_key:
            st.session_state.user_api_key = api_key
            os.environ["GROQ_API_KEY"] = api_key
            st.session_state.api_key_set = bool(api_key)
            # Reset chain to use new key
            st.session_state.rag_chain = None

        if st.session_state.api_key_set:
            st.markdown('<span class="status-badge status-ready">✓ API Key Set</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="status-badge status-error">✗ API Key Missing</span>',
                unsafe_allow_html=True
            )
            st.info("👉 Get a free key at [console.groq.com](https://console.groq.com)", icon="🔗")

        st.divider()

        # Model selector
        st.markdown('<div class="section-header">🤖 Model Settings</div>', unsafe_allow_html=True)
        model_options = [
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        selected_model = st.selectbox(
            "LLM Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_options else 0,
            help="All models run on Groq's free tier",
            key="model_selector",
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.rag_chain = None  # Rebuild with new model

        st.divider()

        # Knowledge Base Section
        st.markdown('<div class="section-header">📚 Knowledge Base</div>', unsafe_allow_html=True)

        if st.session_state.kb_built:
            st.markdown(f"""
            <span class="status-badge status-ready">✓ KB Ready</span>
            <div style="margin-top:0.5rem; font-size:0.82rem; color:#94a3b8;">
                📄 {st.session_state.kb_paper_count} papers · 
                🔢 {st.session_state.kb_doc_count:,} chunks
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Rebuild Knowledge Base", use_container_width=True,
                        help="Re-download and re-index all papers"):
                _reset_and_rebuild()
        else:
            st.markdown(
                '<span class="status-badge status-building">○ Not Built</span>',
                unsafe_allow_html=True
            )
            if st.button(
                "🚀 Build Knowledge Base",
                use_container_width=True,
                type="primary",
                disabled=not st.session_state.api_key_set,
                help="Downloads AI/ML papers from arXiv and indexes them",
            ):
                _build_knowledge_base()

        # Paper list
        with st.expander("📋 Papers in Knowledge Base", expanded=False):
            for pid in ARXIV_PAPER_IDS:
                title = PAPER_TITLES.get(pid, pid)
                st.markdown(f"""
                <div class="tag-pill">arXiv:{pid}</div>
                <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:0.5rem;">{title}</div>
                """, unsafe_allow_html=True)

        st.divider()

        # Memory stats
        st.markdown('<div class="section-header">🧠 Memory Status</div>', unsafe_allow_html=True)
        if st.session_state.memory_manager:
            stats = st.session_state.memory_manager.get_memory_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Window", f"{stats['current_window_turns']}/{stats['window_k']}")
            with col2:
                st.metric("Total Turns", stats["total_turns"])
        else:
            st.caption("No active conversation")

        if st.button("🗑️ Clear Memory", use_container_width=True, help="Reset conversation history"):
            _clear_memory()

        st.divider()

        # Actions
        st.markdown('<div class="section-header">⚡ Actions</div>', unsafe_allow_html=True)

        if st.button("📊 Run Evaluation Suite", use_container_width=True,
                    disabled=not st.session_state.kb_built,
                    help="Evaluate RAG quality on benchmark questions"):
            st.session_state.show_eval = True
            st.rerun()

        if st.session_state.messages and st.session_state.memory_manager:
            export_text = st.session_state.memory_manager.export_as_text()
            st.download_button(
                "💾 Export Chat History",
                data=export_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ─── Knowledge Base Builder ───────────────────────────────────────────────────
def _build_knowledge_base():
    """Download papers, build embeddings, and create vector store."""
    try:
        with st.status("🔄 Building Knowledge Base...", expanded=True) as build_status:
            # ── Step 1: Init vector store ──────────────────────────────────
            st.write("**Step 1/3:** Initializing vector store...")
            vsm = VectorStoreManager()

            documents = []

            # Check if already indexed
            if vsm.is_populated():
                st.write("**Loading existing knowledge base...**")
                vsm.load()
            else:
                # ── Step 2: Download + parse papers ───────────────────────
                st.write("**Step 2/3:** Downloading & parsing arXiv papers from arXiv...")
                loader = ArxivDocumentLoader()

                log_placeholder = st.empty()
                log_lines: list = []

                def progress_cb(paper_id, status, message):
                    log_lines.append(message)
                    log_placeholder.code("\n".join(log_lines[-6:]), language=None)

                documents = loader.load_all(progress_callback=progress_cb)
                log_placeholder.empty()

                if not documents:
                    build_status.update(label="❌ Build failed — no documents loaded", state="error")
                    st.error("❌ No documents were loaded. Check your internet connection.")
                    return

                # ── Step 3: Build embeddings + index ──────────────────────
                st.write(f"**Step 3/3:** Embedding {len(documents):,} chunks into ChromaDB...")
                vsm.build(documents)

            # ── Finalize ──────────────────────────────────────────────────
            stats = vsm.get_stats()
            doc_count = stats.get("document_count", len(documents))
            st.session_state.vsm = vsm
            st.session_state.kb_built = True
            st.session_state.kb_doc_count = doc_count
            st.session_state.kb_paper_count = len(ARXIV_PAPER_IDS)

            # Build RAG chain
            _build_rag_chain(vsm)

            build_status.update(
                label=f"✅ Knowledge base ready — {doc_count:,} chunks indexed!",
                state="complete",
                expanded=False,
            )

    except Exception as exc:
        logger.error("KB build error: %s", exc, exc_info=True)
        st.error(f"❌ Build failed: {exc}")
        return

    time.sleep(1.0)
    st.rerun()


def _reset_and_rebuild():
    """Reset the vector store and rebuild from scratch."""
    vsm = VectorStoreManager()
    vsm.reset()
    st.session_state.vsm = None
    st.session_state.kb_built = False
    st.session_state.rag_chain = None
    st.session_state.memory_manager = None
    _build_knowledge_base()


def _build_rag_chain(vsm: VectorStoreManager = None):
    """Initialize or reinitialize the RAG chain."""
    if vsm is None:
        vsm = st.session_state.vsm
    if vsm is None:
        return

    api_key = st.session_state.user_api_key
    if not api_key:
        return

    os.environ["GROQ_API_KEY"] = api_key

    try:
        memory_manager = ConversationMemoryManager()
        chain = RAGChain(
            vector_store_manager=vsm,
            memory_manager=memory_manager,
            model_name=st.session_state.selected_model,
        )
        st.session_state.rag_chain = chain
        st.session_state.memory_manager = memory_manager
        logger.info("RAG chain initialized with model: %s", st.session_state.selected_model)
    except Exception as exc:
        st.error(f"❌ Failed to initialize RAG chain: {exc}")
        logger.error("Chain init error: %s", exc, exc_info=True)


def _clear_memory():
    """Clear conversation history."""
    if st.session_state.rag_chain:
        st.session_state.rag_chain.clear_memory()
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.session_state.last_scores = []
    st.success("🗑️ Memory cleared!")
    time.sleep(0.5)
    st.rerun()


# ─── Source Document Panel ────────────────────────────────────────────────────
def render_source_panel(source_docs: list, scores: list):
    """Render retrieved source documents with relevance scores."""
    if not source_docs:
        return

    st.markdown("---")
    st.markdown(f"**📎 Retrieved Sources** — {len(source_docs)} chunks")

    for i, doc in enumerate(source_docs[:MAX_SOURCES_DISPLAYED]):
        score = scores[i] if i < len(scores) else 0.0
        score_pct = int(score * 100)

        # Score classification
        if score >= 0.7:
            score_class = "score-high"
            score_label = "High"
        elif score >= 0.4:
            score_class = "score-medium"
            score_label = "Medium"
        else:
            score_class = "score-low"
            score_label = "Low"

        title = doc.metadata.get("title", "Unknown Paper")
        paper_id = doc.metadata.get("paper_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        arxiv_url = doc.metadata.get("arxiv_url", f"https://arxiv.org/abs/{paper_id}")
        preview = doc.page_content[:300].strip()

        st.markdown(f"""
        <div class="source-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.4rem;">
                <div>
                    <div class="source-title">📄 {title}</div>
                    <div style="color:#64748b; font-size:0.75rem;">
                        arXiv:{paper_id} · Chunk #{chunk_idx}
                    </div>
                </div>
                <span class="source-score {score_class}">
                    {score_label} {score_pct}%
                </span>
            </div>
            <div style="color:#cbd5e1; font-size:0.82rem; line-height:1.5;">
                {preview}...
            </div>
            <div style="margin-top:0.4rem;">
                <a href="{arxiv_url}" target="_blank" 
                   style="color:#7c3aed; font-size:0.75rem; text-decoration:none;">
                    🔗 View on arXiv →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Evaluation Panel ─────────────────────────────────────────────────────────
def render_evaluation_panel():
    """Render the RAG evaluation suite results."""
    st.markdown("---")
    st.markdown("## 📊 RAG Evaluation Suite")
    st.caption("Tests the chatbot on benchmark AI/ML questions to measure retrieval quality and answer accuracy.")

    if not st.session_state.kb_built or not st.session_state.rag_chain:
        st.warning("⚠️ Please build the knowledge base and set your API key first.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        n_questions = st.slider("Number of evaluation questions", 1, len(EVAL_QUESTIONS),
                                min(4, len(EVAL_QUESTIONS)), key="eval_n")
    with col2:
        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_eval or st.session_state.eval_results:
        if run_eval:
            with st.spinner("🔄 Running evaluation..."):
                retriever = ContextualRetriever(st.session_state.vsm)
                evaluator = RAGEvaluator(
                    rag_chain=st.session_state.rag_chain,
                    retriever=retriever,
                    eval_questions=EVAL_QUESTIONS[:n_questions],
                )
                progress_bar = st.progress(0)
                status = st.empty()

                def prog_cb(i, total, q):
                    progress_bar.progress((i + 1) / total)
                    status.caption(f"Q{i+1}/{total}: {q[:60]}...")

                results = evaluator.run_evaluation_suite(progress_callback=prog_cb)
                st.session_state.eval_results = results
                progress_bar.empty()
                status.empty()

        results = st.session_state.eval_results
        if not results:
            return

        agg = results.get("aggregate_metrics", {})

        # ── Aggregate metrics ──────────────────────────────────────────────
        st.markdown("### 📈 Aggregate Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, f"{agg.get('avg_latency', 0):.2f}s", "Avg Latency"),
            (c2, f"{agg.get('avg_sources', 0):.1f}", "Avg Sources"),
            (c3, f"{agg.get('avg_relevance_score', 0):.3f}", "Avg Relevance"),
            (c4, f"{agg.get('answer_non_empty_rate', 0)*100:.0f}%", "Answer Rate"),
            (c5, f"{agg.get('avg_unique_papers', 0):.1f}", "Unique Papers"),
        ]
        for col, value, label in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📝 Per-Question Results")
        for i, res in enumerate(results.get("individual_results", []), 1):
            q = res.get("question", "")
            a = res.get("answer", "")
            latency = res.get("latency_seconds", 0)
            n_src = res.get("num_sources", 0)
            rel_score = res.get("retrieval_metrics", {}).get("avg_relevance_score", 0)

            with st.expander(f"Q{i}: {q[:70]}...", expanded=(i == 1)):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Latency", f"{latency:.2f}s")
                col_b.metric("Sources", n_src)
                col_c.metric("Relevance", f"{rel_score:.3f}")

                st.markdown("**Answer:**")
                st.markdown(a)

                sources = res.get("sources", [])
                if sources:
                    st.markdown("**Sources:**")
                    for src in sources:
                        st.markdown(f"""
                        <div class="tag-pill">📄 {src['title'][:40]} 
                        — Score: {src['score']:.3f}</div>
                        """, unsafe_allow_html=True)

        # Download report
        evaluator_dummy = RAGEvaluator(
            rag_chain=st.session_state.rag_chain,
            retriever=ContextualRetriever(st.session_state.vsm),
        )
        report_text = evaluator_dummy.generate_report(results)
        st.download_button(
            "📥 Download Evaluation Report (Markdown)",
            data=report_text,
            file_name="rag_evaluation_report.md",
            mime="text/markdown",
        )

    if st.button("✖ Close Evaluation Panel"):
        st.session_state.show_eval = False
        st.rerun()


# ─── Main Chat Interface ──────────────────────────────────────────────────────
def render_chat():
    """Render the main chat interface."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rag-header">
        <h1>🧠 RAG Research Assistant</h1>
        <p>Context-Aware Chatbot powered by <strong>LangChain</strong> · 
           <strong>Groq Llama 3.1</strong> · <strong>ChromaDB</strong> · 
           <strong>HuggingFace Embeddings</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kb_status = "✅ Ready" if st.session_state.kb_built else "○ Not Built"
        kb_color = "#10b981" if st.session_state.kb_built else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{kb_color}; font-size:1rem;">{kb_status}</div>
            <div class="metric-label">Knowledge Base</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.kb_doc_count:,}</div>
            <div class="metric-label">Indexed Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        turns = st.session_state.memory_manager.get_total_turns() if st.session_state.memory_manager else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{turns}</div>
            <div class="metric-label">Conversation Turns</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        model_short = st.session_state.selected_model.split("-")[0].upper()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1rem;">{model_short}</div>
            <div class="metric-label">Active Model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not st.session_state.api_key_set:
        st.warning("⚠️ **API Key Required** — Enter your Groq API key in the sidebar to start chatting.", icon="🔑")

    if not st.session_state.kb_built:
        st.info(
            "📚 **Knowledge base not built.** Click **'Build Knowledge Base'** in the sidebar "
            "to download AI/ML papers from arXiv and index them. This takes ~2-3 minutes.",
            icon="🚀"
        )

    # Ensure RAG chain is ready when KB is built
    if st.session_state.kb_built and st.session_state.rag_chain is None and st.session_state.api_key_set:
        if st.session_state.vsm is None:
            vsm = VectorStoreManager()
            vsm.load()
            st.session_state.vsm = vsm
        _build_rag_chain()

    # ── Chat history display ───────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#64748b;">
                <div style="font-size:3rem; margin-bottom:1rem;">💬</div>
                <div style="font-size:1.1rem; font-weight:600; color:#94a3b8;">
                    Ask anything about AI & Machine Learning research
                </div>
                <div style="font-size:0.85rem; margin-top:0.5rem;">
                    Try: "What is the transformer architecture?" or "How does RAG improve accuracy?"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
                    st.markdown(content)

    # ── Sources from last query ────────────────────────────────────────────────
    if st.session_state.last_sources:
        render_source_panel(
            st.session_state.last_sources,
            st.session_state.last_scores,
        )

    # ── Evaluation panel ──────────────────────────────────────────────────────
    if st.session_state.show_eval:
        render_evaluation_panel()

    # ── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state.messages and st.session_state.kb_built:
        st.markdown("**💡 Suggested Questions:**")
        suggested = [
            "What is the transformer architecture and how does self-attention work?",
            "How does Retrieval-Augmented Generation (RAG) improve LLM accuracy?",
            "Explain chain-of-thought prompting and why it helps LLMs reason better.",
            "What are the key differences between LLaMA and LLaMA 2?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggested):
            with cols[i % 2]:
                if st.button(f"💬 {q[:60]}...", key=f"sugg_{i}", use_container_width=True):
                    _handle_question(q)

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask a question about AI/ML research...",
        disabled=not (st.session_state.kb_built and st.session_state.api_key_set),
        key="chat_input",
    ):
        _handle_question(prompt)


def _handle_question(question: str):
    """Process a user question through the RAG pipeline and display the response."""

    # Ensure chain is ready
    if not st.session_state.rag_chain:
        if st.session_state.kb_built and st.session_state.api_key_set:
            _build_rag_chain()
        else:
            st.error("Please build the knowledge base and set your API key first.")
            return

    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔍 Searching knowledge base and generating response..."):
            result = st.session_state.rag_chain.ask(question)

        answer = result.get("answer", "I could not generate a response.")
        source_docs = result.get("source_documents", [])
        scores = result.get("scores", [])

        st.markdown(answer)

        # Show inline source count
        if source_docs:
            st.caption(f"📎 Retrieved {len(source_docs)} sources · Model: {result.get('model', 'unknown')}")

    # Update state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = source_docs
    st.session_state.last_scores = scores


# ─── Main Entry Point ─────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()
    render_chat()
    # Evaluation panel renders below the chat when triggered from sidebar
    if st.session_state.get("show_eval"):
        render_evaluation_panel()


if __name__ == "__main__":
    main()
