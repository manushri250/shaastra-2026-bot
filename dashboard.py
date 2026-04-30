# Run with: python -m streamlit run dashboard.py

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

from app import (
    EMBEDDING_MODEL,
    PDF_PATH,
    build_index,
    build_query_context,
    extract_pdf_text,
    generate_answer,
    load_llm,
    parse_tree,
    retrieve,
)


SAMPLE_QUESTIONS = [
    "What events are on January 4?",
    "Where is Robowars held?",
    "Tell me about AI workshops",
    "What time does the Comedy Night start?",
    "Who are the spotlight lecture speakers?",
]


st.set_page_config(
    page_title="Shaastra 2026 Assistant",
    page_icon="\U0001f916",
    layout="wide",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --page: #f7f9fc;
            --panel: #ffffff;
            --line: #dce5f2;
            --text: #172033;
            --muted: #617089;
            --orange: #e86f1f;
            --orange-soft: #fff3ea;
            --blue: #2563eb;
            --blue-soft: #eff6ff;
        }

        html, body, [class*="css"] {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
        }

        .stApp {
            background: var(--page);
        }

        .block-container {
            max-width: 980px;
            padding-top: 1.25rem;
            padding-bottom: 5rem;
        }

        section[data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        .logo {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--text);
            margin: 0.2rem 0 0.35rem;
            letter-spacing: 0;
        }

        .logo-accent {
            color: var(--orange);
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .sidebar-section {
            color: var(--orange);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 1rem 0 0.55rem;
        }

        .main-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.9rem 1rem;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            margin-bottom: 0.45rem;
            box-shadow: 0 8px 24px rgba(23, 32, 51, 0.07);
        }

        .main-header h1 {
            color: var(--text);
            font-size: 1.75rem;
            line-height: 1.1;
            margin: 0;
            letter-spacing: 0;
        }

        .tagline {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 500;
            margin-top: 0.3rem;
        }

        .status-pill {
            color: #174ea6;
            background: var(--blue-soft);
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 0.4rem 0.7rem;
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .empty-state {
            border: 1px dashed #cbd5e1;
            background: var(--panel);
            border-radius: 8px;
            padding: 1.35rem;
            text-align: center;
            color: var(--muted);
            margin: 0.7rem auto 0.3rem;
            max-width: 620px;
        }

        .empty-state strong {
            display: block;
            color: var(--text);
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }

        div[data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
            margin: 0.45rem 0;
            padding: 0.25rem 0.4rem;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            color: var(--text);
            line-height: 1.55;
        }

        div[data-testid="stChatInput"] {
            background: var(--page);
        }

        div[data-testid="stChatInput"] textarea {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #ffffff;
            color: var(--text);
            box-shadow: 0 6px 20px rgba(23, 32, 51, 0.08);
        }

        div[data-testid="stButton"] button {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: #ffffff;
            color: var(--text);
            font-weight: 600;
            min-height: 2.45rem;
        }

        div[data-testid="stButton"] button:hover {
            border-color: var(--orange);
            background: var(--orange-soft);
            color: var(--text);
        }

        hr {
            border-color: var(--line);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    text = extract_pdf_text(PDF_PATH)
    root = parse_tree(text)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    nodes, embeddings = build_index(root, embedder)
    llm = load_llm()
    return root, embedder, nodes, embeddings, llm


def ask_bot(query: str) -> str:
    root, embedder, nodes, embeddings, llm = load_pipeline()
    retrieved_nodes = retrieve(query, embedder, nodes, embeddings, top_k=3)
    context = build_query_context(query, root, retrieved_nodes)
    return generate_answer(query, context, llm)


def set_query(question: str) -> None:
    st.session_state.pending_query = question


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:
    st.markdown(
        '<div class="logo">Shaastra <span class="logo-accent">2026</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-copy">Your AI guide to Shaastra 2026, IIT Madras</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Try asking</div>', unsafe_allow_html=True)
    for question in SAMPLE_QUESTIONS:
        st.button(
            question,
            key=f"sample_{question}",
            on_click=set_query,
            args=(question,),
            use_container_width=True,
        )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()


st.markdown(
    """
    <div class="main-header">
        <div>
            <h1>Shaastra 2026 Assistant</h1>
            <div class="tagline">Explore events, workshops, venues, lectures, and schedules.</div>
        </div>
        <div class="status-pill">Local RAG assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Start a conversation</strong>
            Ask about events, venues, schedules, workshops, lectures, or competitions.
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "\U0001f464" if message["role"] == "user" else "\U0001f916"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


typed_query = st.chat_input("Ask about Shaastra 2026...")
active_query = None
if typed_query and typed_query.strip():
    active_query = typed_query.strip()
elif st.session_state.pending_query:
    active_query = st.session_state.pending_query.strip()
    st.session_state.pending_query = None

if active_query:
    with st.chat_message("user", avatar="\U0001f464"):
        st.markdown(active_query)

    with st.chat_message("assistant", avatar="\U0001f916"):
        with st.spinner("Thinking..."):
            bot_answer = ask_bot(active_query)
        st.markdown(bot_answer)

    st.session_state.messages.append({"role": "user", "content": active_query})
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
