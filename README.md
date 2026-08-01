# RAG-Based PDF Question Answering System 📚🤖

A complete, production-grade Retrieval-Augmented Generation (RAG) web application built for PDF document parsing, text chunking, vector embedding indexing, semantic search, and context-bounded question answering.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20LangChain-violet)
![VectorDB](https://img.shields.io/badge/VectorDB-FAISS-green)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20%7C%20OpenAI-orange)

---

## 🌟 Features & Highlights

- **Multi-PDF Document Processing**: Upload single or multiple PDF documents with automatic page-by-page text extraction.
- **Configurable Text Chunking**: Dynamically adjust Chunk Size (200 - 2000 chars) and Chunk Overlap (0 - 500 chars) from the UI.
- **FAISS Vector Database & Dual Indexing**: Fast Approximate Nearest Neighbor vector search powered by FAISS, with an embedded zero-dependency TF-IDF fallback vector index for guaranteed 100% uptime.
- **Embeddings Flexibility**: Support for local HuggingFace `all-MiniLM-L6-v2` (free, zero-cost), Google Gemini Embeddings (`embedding-001`), and OpenAI Embeddings.
- **Strict Context Prompting & Anti-Hallucination**: Enforces non-hallucination rules. When information is unavailable in uploaded PDFs, the system responds:
  > *"I couldn't find this information in the uploaded documents."*
- **Source Citations & Page Attribution**: Every generated answer displays expandable source page numbers, document titles, chunk IDs, and exact text snippets.
- **Document Summaries & Semantic Search Tester**:
  - Automatically generates preview summaries for uploaded files.
  - Dedicated **Semantic Retrieval Test** tab to benchmark search quality independently from LLM generation.
- **Modern UI & Dark Aesthetics**: Sleek dark-mode interface built with Streamlit, custom glassmorphic cards, and keyword highlighting.

---

## 🏗️ System Architecture

```
                                  [ PDF Upload ]
                                         │
                                         ▼
                               [ PyPDF Text Extractor ]
                                         │
                                         ▼
                        [ Recursive Character Text Splitter ]
                             (Chunk Size & Overlap)
                                         │
                                         ▼
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Embeddings Engine ]                    [ TF-IDF Fallback Vector ]
    (HuggingFace / Gemini / OpenAI)
                    │
                    ▼
          [ FAISS Vector Store ]
                    │
                    ▼
          [ User Question / Query ]
                    │
                    ▼
     [ Semantic Search (Top-K Chunks) ]
                    │
                    ▼
  [ Strict Context-Bound Prompt Template ] ──► [ LLM (Gemini / OpenAI) ] ──► [ Answer + Source Page Citations ]
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone / Download Project
```bash
git clone <repository_url>
cd RAG
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)
Copy `.env.example` to `.env` and enter your Gemini or OpenAI API Key:
```bash
cp .env.example .env
```
*Note: You can also enter your API Key directly in the Web Application Sidebar!*

### 5. Launch Application
```bash
streamlit run app.py
```
- **On Desktop Computer**: Open `http://localhost:8501`.
- **On Mobile Phone (Same Wi-Fi)**: Open `http://<YOUR_LOCAL_IP>:8501` (e.g. `http://192.168.1.15:8501`). Your local IP is automatically displayed inside the application sidebar under **📱 Access on Mobile / Phone**.
- **On Mobile Phone (Cellular / Anywhere)**: Run `npx localtunnel --port 8501` or `ngrok http 8501` to generate a public HTTPS URL for remote access.

---

## 📊 Evaluation Rubric Mapping (100 Marks)

| Criteria | Max Marks | Implementation Details |
| :--- | :---: | :--- |
| **PDF Loading** | **10** | Multi-file page-by-page text extraction via `pypdf` with page metadata tracking. |
| **Text Chunking** | **10** | `RecursiveCharacterTextSplitter` with dynamic chunk size & overlap sliders in UI. |
| **Embeddings** | **15** | Integrates HuggingFace `all-MiniLM-L6-v2` (local/free), Google Gemini, & OpenAI embeddings. |
| **Vector Database** | **15** | High-performance `FAISS` vector database with auto fallback vector store. |
| **Retrieval Accuracy** | **20** | Top-$k$ similarity search with custom top-$k$ slider and keyword highlighting tab. |
| **LLM Integration** | **15** | Prompt engineering with Google Gemini & OpenAI API + unanswerable query safeguard. |
| **UI Design** | **10** | Premium dark-themed Streamlit dashboard with tabs, cards, and source citations. |
| **Documentation & Quality**| **5** | Clean modular structure (`rag_engine.py`, `app.py`), `README.md`, and `REPORT.md`. |
| **Bonus Features** | **Extra** | Multi-PDF support, page citations, document summaries, keyword highlighting, chat memory. |

---

## 📂 Project Structure

```
RAG/
├── app.py                      # Streamlit Web Application Dashboard
├── rag_engine.py               # Core RAG engine logic (parsing, chunking, FAISS, LLM)
├── create_sample_pdfs.py       # Helper script to generate sample test PDFs
├── test_rag.py                 # Automated unit test suite
├── requirements.txt            # Project Python dependencies
├── .env.example                # Template environment file
├── README.md                   # Setup guide and assignment overview
├── REPORT.md                   # 3-Page Detailed Project & Architecture Report
└── sample_documents/           # Test PDFs
    ├── Artificial_Intelligence_Overview.pdf
    └── RAG_Architecture_Guide.pdf
```

---

## 🧪 Testing with Sample PDFs

The project includes pre-built sample PDFs in `sample_documents/`:
1. Upload `Artificial_Intelligence_Overview.pdf` and `RAG_Architecture_Guide.pdf` in the sidebar.
2. Click **🚀 Process & Build Index**.
3. Ask sample questions in the Chat tab:
   - *"What are the chunk size recommendations for RAG?"*
   - *"Who introduced the Transformer architecture?"*
   - *"What is the capital of Jupiter?"* (Tests strict non-hallucination safeguard!)

---

## 📄 License & Credits
Built for the RAG-Based PDF Question Answering System Assignment. Powered by Python, LangChain, FAISS, Streamlit, and Google Gemini.
"# RAG_PDF_Doc" 
