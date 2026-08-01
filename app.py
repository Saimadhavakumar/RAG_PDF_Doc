import os
import sys
import warnings

# Suppress environment warnings and deprecation logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from rag_engine import RAGEngine, FAISS_AVAILABLE
from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

# Page Configuration & Custom Aesthetics
st.set_page_config(
    page_title="RAG PDF Question Answering System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

import socket
import re

# Helper function to get Host IP for mobile phone connectivity
def get_local_ip():
    """Retrieves local IPv4 address of the host for mobile connectivity."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Dark Mode & Modern CSS Styling (Mobile Responsive)
CUSTOM_CSS = """
<style>
    /* Hide unneeded Streamlit header elements while preserving mobile sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    button[title="View code"] {display: none !important;}
    a[href*="github.com"] {display: none !important;}

    /* Preserve header container as transparent so mobile sidebar toggle button is visible & clickable */
    header[data-testid="stHeader"], div[data-testid="stHeader"] {
        background: transparent !important;
    }
    /* ChatGPT-Style Movable Sidebar Toggle (When Collapsed) */
    div[data-testid="collapsedControl"] {
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 999999 !important;
        background: rgba(30, 41, 59, 0.9) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 10px !important;
        padding: 6px 10px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), 0 0 12px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    div[data-testid="collapsedControl"]:hover {
        background: rgba(99, 102, 241, 0.3) !important;
        border-color: #a855f7 !important;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.4) !important;
        transform: translateY(-1px) scale(1.04);
    }
    div[data-testid="collapsedControl"] button {
        color: #818cf8 !important;
    }
    div[data-testid="collapsedControl"] button:hover {
        color: #ffffff !important;
    }

    /* ChatGPT-Style Sidebar Container & Close Button (When Expanded) */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s ease !important;
    }

    button[data-testid="stSidebarCollapseButton"], 
    section[data-testid="stSidebar"] button[kind="header"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover, 
    section[data-testid="stSidebar"] button[kind="header"]:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
    }

    /* Dark glassmorphism & accent theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .source-box {
        background-color: #1e293b;
        border-left: 4px solid #6366f1;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        word-break: break-word;
    }
    .highlight-keyword {
        background-color: #f59e0b;
        color: #000000;
        font-weight: bold;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
    }
    .stChatMessage {
        border-radius: 12px;
    }

    /* Mobile Phone Responsive Adjustments (<768px) */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 1.5rem !important;
            margin-top: 0.5rem;
        }
        .sub-header {
            font-size: 0.88rem !important;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 0.75rem 0.9rem !important;
            margin-bottom: 0.75rem !important;
        }
        .metric-card h2 {
            font-size: 1.2rem !important;
        }
        .metric-card h4 {
            font-size: 0.8rem !important;
        }
        .stButton > button {
            width: 100% !important;
            min-height: 44px !important;
            font-size: 0.95rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            overflow-x: auto !important;
            white-space: nowrap !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 8px 12px !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []


def highlight_keywords(text: str, query: str) -> str:
    """Highlights keywords from the query inside text snippet."""
    if not query.strip():
        return text
    words = [w for w in query.split() if len(w) > 3]
    for w in set(words):
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        text = pattern.sub(f"<span class='highlight-keyword'>{w}</span>", text)
    return text


# Header Section
st.markdown('<div class="main-header">📚 RAG PDF Question Answering System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload multi-page PDFs, process chunks, generate vector embeddings & query context-bounded answers</div>', unsafe_allow_html=True)

# Sidebar Configuration Panel
local_ip = get_local_ip()
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    with st.expander("📱 Access on Mobile / Phone", expanded=False):
        st.markdown(f"""
        **1. Local Wi-Fi Access:**
        Connect your phone to the same Wi-Fi network and open:
        ```text
        http://{local_ip}:8501
        ```
        **2. Internet Access (Cellular / Anywhere):**
        Run this command in terminal to generate a secure public link:
        ```bash
        npx localtunnel --port 8501
        ```
        *or* `ngrok http 8501`
        """)
        
    st.subheader("1. LLM & Embeddings")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=["groq", "gemini", "openai", "extractive_only"],
        format_func=lambda x: "⚡ Groq Cloud (Llama 3 - Fast)" if x == "groq" else ("Google Gemini AI" if x == "gemini" else ("OpenAI GPT" if x == "openai" else "Extractive Context Only (No API Key)"))
    )
    
    api_key_input = ""
    if llm_provider == "groq":
        default_key = os.environ.get("GROQ_API_KEY") or ""
        api_key_input = st.text_input("Groq API Key", value=default_key, type="password", help="Enter Groq API Key")
    elif llm_provider == "gemini":
        default_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        api_key_input = st.text_input("Gemini API Key", value=default_key, type="password", help="Enter Google Gemini API Key")
    elif llm_provider == "openai":
        default_key = os.environ.get("OPENAI_API_KEY") or ""
        api_key_input = st.text_input("OpenAI API Key", value=default_key, type="password", help="Enter OpenAI API Key")

    st.subheader("2. Text Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)

    st.subheader("3. Retrieval Settings")
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4, step=1)
    vector_backend = st.radio("Vector Index Backend", options=["FAISS Vector DB" if FAISS_AVAILABLE else "FAISS (Not Available)", "Fallback TF-IDF Index"], index=0)

    st.divider()

    st.subheader("📁 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple PDF documents to process"
    )

    if st.button("🚀 Process & Build Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
        else:
            with st.spinner("Parsing PDFs and generating vector embeddings..."):
                pdf_tuples = [(f.name, f.read()) for f in uploaded_files]
                engine = st.session_state.rag_engine
                
                # Step 1: Extract Text
                engine.extract_text_from_pdfs(pdf_tuples)
                
                # Step 2: Chunk Text
                chunks = engine.process_and_chunk_text(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.session_state.last_chunks = chunks

                # Step 3: Build Vector Index
                emb_provider = "gemini" if (llm_provider == "gemini" and api_key_input) else "huggingface"
                index_msg = engine.build_vector_store(provider=emb_provider, api_key=api_key_input)

                st.session_state.indexed = True
                st.success(index_msg)

    st.divider()
    if st.button("🗑️ Clear Chat & Reset", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Display Main Content Dashboard
if st.session_state.indexed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#94a3b8;">Uploaded Documents</h4>
            <h2 style="margin:0; color:#6366f1;">{len(st.session_state.rag_engine.raw_documents)} pages</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#94a3b8;">Total Text Chunks</h4>
            <h2 style="margin:0; color:#a855f7;">{len(st.session_state.rag_engine.chunks)} chunks</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#94a3b8;">Vector Database</h4>
            <h2 style="margin:0; color:#10b981;">{"FAISS Active" if st.session_state.rag_engine.vector_store else "TF-IDF Active"}</h2>
        </div>
        """, unsafe_allow_html=True)

# Tabs Navigation
tab_chat, tab_inspector, tab_search, tab_summaries = st.tabs([
    "💬 Chat Assistant", 
    "🔍 Chunk Inspector", 
    "🎯 Semantic Retrieval Test", 
    "📑 Document Summaries"
])

# TAB 1: Chat Assistant
with tab_chat:
    st.caption("Ask questions strictly based on your uploaded PDFs. Sources & page numbers are automatically cited.")

    # Render previous conversation history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📖 View Source Citations"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source:</strong> {src['source']} | <strong>Page:</strong> {src['page']} | <strong>Chunk #{src['chunk_id']}</strong><br/>
                            <em>"{src['snippet']}"</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the PDF content...")

    if user_query:
        if not st.session_state.indexed:
            st.warning("⚠️ Please upload and click 'Process & Build Index' in the sidebar first.")
        else:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with st.spinner("Retrieving top-k chunks & generating answer..."):
                    engine = st.session_state.rag_engine
                    
                    # 1. Retrieve relevant chunks
                    retrieved_chunks = engine.retrieve_relevant_chunks(user_query, top_k=top_k)
                    
                    # 2. Generate LLM Answer
                    res = engine.generate_answer(
                        question=user_query,
                        retrieved_chunks=retrieved_chunks,
                        llm_provider=llm_provider,
                        api_key=api_key_input
                    )
                    
                    answer_text = res["answer"]
                    sources = res["sources"]

                    st.markdown(answer_text)

                    if sources:
                        with st.expander("📖 View Source Citations"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source:</strong> {src['source']} | <strong>Page:</strong> {src['page']} | <strong>Chunk #{src['chunk_id']}</strong><br/>
                                    <em>"{src['snippet']}"</em>
                                </div>
                                """, unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })

# TAB 2: Chunk Inspector
with tab_inspector:
    st.subheader("Inspect Generated Text Chunks")
    if not st.session_state.indexed or not st.session_state.last_chunks:
        st.info("Upload and process PDFs to inspect text chunks.")
    else:
        chunks = st.session_state.last_chunks
        st.write(f"Total chunks created: **{len(chunks)}**")
        chunk_num = st.number_input("Select Chunk Index", min_value=1, max_value=len(chunks), value=1)
        target_chunk = chunks[chunk_num - 1]
        
        st.markdown(f"**Chunk #{target_chunk.metadata['chunk_id']}** | **File:** `{target_chunk.metadata['source']}` | **Page:** `{target_chunk.metadata['page']}`")
        st.text_area("Chunk Content", value=target_chunk.page_content, height=250)

# TAB 3: Semantic Retrieval Test (Rubric Evaluation)
with tab_search:
    st.subheader("Semantic Similarity Search Tester")
    st.caption("Test the retriever accuracy independently from the LLM.")
    
    test_query = st.text_input("Enter test query to search vector store", value="")
    if test_query:
        if not st.session_state.indexed:
            st.warning("Please index documents first.")
        else:
            engine = st.session_state.rag_engine
            results = engine.retrieve_relevant_chunks(test_query, top_k=top_k)
            st.write(f"Top-{len(results)} Retrieved Chunks:")
            for idx, r in enumerate(results, 1):
                st.markdown(f"### Result {idx} (Similarity Score: `{r['score']}`)")
                st.markdown(f"**Document:** {r['source']} | **Page:** {r['page']} | **Chunk ID:** {r['chunk_id']}")
                highlighted_content = highlight_keywords(r['content'], test_query)
                st.markdown(f'<div class="source-box">{highlighted_content}</div>', unsafe_allow_html=True)

# TAB 4: Document Summaries
with tab_summaries:
    st.subheader("Document Summaries & Overview")
    if not st.session_state.indexed:
        st.info("Upload and process documents to view summaries.")
    else:
        engine = st.session_state.rag_engine
        unique_docs = sorted(list(set(d["source"] for d in engine.raw_documents)))
        for doc_name in unique_docs:
            summary = engine.generate_document_summary(doc_name)
            st.markdown(summary)
            st.divider()
