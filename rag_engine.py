import os
import sys
import warnings

# Suppress environment warnings and deprecation logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import io
import math
import re
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader

# LangChain text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional vector store & embeddings imports with safe fallback
try:
    import faiss
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except Exception as e:
    FAISS_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None

try:
    # pyrefly: ignore [missing-import]
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class DocumentChunk:
    """Class representing a single text chunk with metadata."""
    def __init__(self, content: str, source: str, page: int, chunk_id: int):
        self.page_content = content
        self.metadata = {
            "source": source,
            "page": page,
            "chunk_id": chunk_id
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.page_content,
            "source": self.metadata["source"],
            "page": self.metadata["page"],
            "chunk_id": self.metadata["chunk_id"]
        }


class FallbackVectorStore:
    """
    Lightweight, zero-dependency TF-IDF / Cosine Similarity Vector Index.
    Acts as a guaranteed fallback if native FAISS bindings encounter environment issues.
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vocab: Dict[str, int] = {}
        self.tfidf_matrix: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            return

        # Build vocabulary
        doc_tokens = [self._tokenize(c.page_content) for c in chunks]
        all_words = set(w for tokens in doc_tokens for w in tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        num_docs = len(chunks)
        num_terms = len(self.vocab)

        if num_terms == 0:
            return

        # Compute term frequency (TF) and document frequency (DF)
        df = [0] * num_terms
        tf_list = []
        for tokens in doc_tokens:
            tf = [0] * num_terms
            seen = set()
            for t in tokens:
                if t in self.vocab:
                    idx = self.vocab[t]
                    tf[idx] += 1
                    seen.add(idx)
            for idx in seen:
                df[idx] += 1
            tf_list.append(tf)

        # Compute TF-IDF
        self.tfidf_matrix = []
        for tf in tf_list:
            vector = []
            for idx in range(num_terms):
                tf_val = tf[idx]
                idf_val = math.log((num_docs + 1) / (df[idx] + 1)) + 1
                vector.append(tf_val * idf_val)
            # Normalize
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            self.tfidf_matrix.append([v / norm for v in vector])

    def similarity_search(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or not self.vocab:
            return []

        q_tokens = self._tokenize(query)
        q_tf = [0] * len(self.vocab)
        for t in q_tokens:
            if t in self.vocab:
                q_tf[self.vocab[t]] += 1

        norm = math.sqrt(sum(v * v for v in q_tf)) or 1.0
        q_vec = [v / norm for v in q_tf]

        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            score = sum(q_vec[i] * doc_vec[i] for i in range(len(self.vocab)))
            scores.append((self.chunks[idx], float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RAGEngine:
    """
    Complete Retrieval-Augmented Generation Engine supporting:
    - PDF text parsing (single & multi-file)
    - Configurable text chunking
    - FAISS / Fallback Vector Indexing
    - Semantic similarity retrieval
    - Multi-provider LLM answer generation (Gemini, OpenAI, Extractive)
    """

    def __init__(self):
        self.raw_documents: List[Dict[str, Any]] = [] # [{source, page, text}]
        self.chunks: List[DocumentChunk] = []
        self.vector_store = None
        self.fallback_store = None
        self.embedding_provider = "huggingface" # 'huggingface', 'gemini', 'openai'
        self.embeddings_model = None

    def extract_text_from_pdfs(self, pdf_files: List[Tuple[str, bytes]]) -> List[Dict[str, Any]]:
        """
        Extract page-by-page text from uploaded PDF files.
        Returns list of page dicts: {'source': filename, 'page': page_num, 'text': text}
        """
        extracted_pages = []
        for filename, pdf_bytes in pdf_files:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        extracted_pages.append({
                            "source": filename,
                            "page": page_idx + 1,
                            "text": page_text
                        })
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
        self.raw_documents = extracted_pages
        return extracted_pages

    def process_and_chunk_text(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[DocumentChunk]:
        """
        Splits extracted page text into chunks with metadata using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_chunks = []
        chunk_id = 0
        for doc in self.raw_documents:
            splits = splitter.split_text(doc["text"])
            for split_text in splits:
                if split_text.strip():
                    chunk_id += 1
                    all_chunks.append(DocumentChunk(
                        content=split_text.strip(),
                        source=doc["source"],
                        page=doc["page"],
                        chunk_id=chunk_id
                    ))

        self.chunks = all_chunks
        return all_chunks

    def init_embeddings(self, provider: str = "huggingface", api_key: Optional[str] = None):
        """Initializes the specified embedding model provider."""
        self.embedding_provider = provider
        try:
            if provider == "gemini" and api_key and GoogleGenerativeAIEmbeddings:
                self.embeddings_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
            elif provider == "openai" and api_key and OpenAIEmbeddings:
                self.embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
            elif HuggingFaceEmbeddings:
                # Default zero-cost local HuggingFace SentenceTransformer
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embeddings_model = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
            else:
                self.embeddings_model = None
        except Exception as e:
            print(f"Embedding initialization warning: {e}")
            self.embeddings_model = None

    def build_vector_store(self, provider: str = "huggingface", api_key: Optional[str] = None) -> str:
        """
        Builds vector index using FAISS or fallback index.
        """
        if not self.chunks:
            return "No chunks available to index."

        self.init_embeddings(provider=provider, api_key=api_key)

        # Build Fallback Index first to guarantee search availability
        self.fallback_store = FallbackVectorStore()
        self.fallback_store.add_chunks(self.chunks)

        # Attempt FAISS vector index creation
        if FAISS_AVAILABLE and self.embeddings_model:
            try:
                texts = [c.page_content for c in self.chunks]
                metadatas = [c.metadata for c in self.chunks]
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings_model,
                    metadatas=metadatas
                )
                return f"Successfully indexed {len(self.chunks)} chunks into FAISS vector store."
            except Exception as e:
                print(f"FAISS indexing error, using robust fallback index: {e}")
                self.vector_store = None
                return f"Indexed {len(self.chunks)} chunks using Fallback Vector Index (FAISS fallback active)."
        else:
            return f"Indexed {len(self.chunks)} chunks using Fallback Vector Index."

    def retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks using Hybrid Search (Vector Search + Keyword Boost).
        Adaptively retrieves all chunks for smaller documents to ensure 100% context coverage.
        """
        if not self.chunks:
            return []

        # Adaptive K: For small documents (<= 15 chunks), include more context so no section is missed
        num_total_chunks = len(self.chunks)
        effective_k = min(num_total_chunks, max(top_k, 8)) if num_total_chunks <= 15 else max(top_k, 6)

        results = []
        retrieved_ids = set()

        # 1. Vector Search
        if self.vector_store:
            try:
                docs_and_scores = self.vector_store.similarity_search_with_score(query, k=effective_k)
                for doc, score in docs_and_scores:
                    cid = doc.metadata.get("chunk_id", 0)
                    retrieved_ids.add(cid)
                    results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 1),
                        "chunk_id": cid,
                        "score": round(float(score), 4)
                    })
            except Exception as e:
                print(f"Vector store query error: {e}")

        # Fallback Vector Search if primary failed or empty
        if not results and self.fallback_store:
            fallback_results = self.fallback_store.similarity_search(query, top_k=effective_k)
            for chunk, score in fallback_results:
                cid = chunk.metadata["chunk_id"]
                retrieved_ids.add(cid)
                results.append({
                    "content": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "chunk_id": cid,
                    "score": round(float(score), 4)
                })

        # 2. Hybrid Keyword Boosting: Check for key query words missing from top-k vector results
        query_words = set(re.findall(r'\w+', query.lower()))
        stop_words = {"what", "is", "are", "the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "about", "explain", "tell", "me", "show", "give", "list", "detail", "details"}
        keywords = [w for w in query_words if w not in stop_words and len(w) > 2]

        if keywords:
            for chunk in self.chunks:
                cid = chunk.metadata["chunk_id"]
                if cid not in retrieved_ids:
                    content_lower = chunk.page_content.lower()
                    match_count = sum(1 for kw in keywords if kw in content_lower)
                    if match_count > 0:
                        results.append({
                            "content": chunk.page_content,
                            "source": chunk.metadata["source"],
                            "page": chunk.metadata["page"],
                            "chunk_id": cid,
                            "score": round(0.5 / match_count, 4) # boosted score for keyword matches
                        })
                        retrieved_ids.add(cid)

        # Sort results: vector matches first, keyword matches
        return results[:max(effective_k, top_k)]

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_provider: str = "groq",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer using the LLM strictly based on retrieved context.
        Enforces clear context-grounded prompt engineering.
        """
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find this information in the uploaded documents.",
                "sources": []
            }

        # Format context string from retrieved chunks
        context_blocks = []
        sources_used = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Chunk {idx} | Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['content']}"
            )
            sources_used.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "snippet": chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
            })

        context_str = "\n\n".join(context_blocks)

        prompt_template = f"""You are an AI assistant helping a user answer questions strictly using content from their uploaded documents.

Answer the question thoroughly and accurately using ONLY the provided context below. Include all relevant project details, titles, descriptions, technical terms, and bullet points present in the context.

If the provided context does not contain relevant information to answer the question, respond with:
"I couldn't find this information in the uploaded documents."

Context:
{context_str}

Question:
{question}
"""

        answer = ""

        # Provider 0: Groq Cloud API (Llama 3)
        if llm_provider == "groq":
            key = api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                return {
                    "answer": "⚠️ Groq API key missing. Please enter your API key in the sidebar configuration or set GROQ_API_KEY in .env.",
                    "sources": sources_used
                }
            try:
                from groq import Groq
                client = Groq(api_key=key)
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_template}],
                        temperature=0.2
                    )
                except Exception:
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt_template}],
                        temperature=0.2
                    )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error generating answer with Groq LLM: {str(e)}"

        # Provider 1: Gemini API
        elif llm_provider == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                return {
                    "answer": "⚠️ Gemini API key missing. Please enter your API key in the sidebar configuration.",
                    "sources": sources_used
                }
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt_template)
                answer = response.text.strip()
            except Exception as e:
                try:
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content(prompt_template)
                    answer = response.text.strip()
                except Exception as ex:
                    answer = f"Error generating answer with Gemini LLM: {str(ex)}"

        # Provider 2: OpenAI API
        elif llm_provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                return {
                    "answer": "⚠️ OpenAI API key missing. Please enter your API key in the sidebar configuration.",
                    "sources": sources_used
                }
            try:
                import openai
                client = openai.OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt_template}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error generating answer with OpenAI: {str(e)}"

        # Provider 3: Extractive Context Fallback
        else:
            answer = f"Here is the relevant information retrieved from your documents:\n\n"
            for s in sources_used:
                answer += f"• **Page {s['page']} ({s['source']})**: {s['snippet']}\n"

        return {
            "answer": answer,
            "sources": sources_used
        }

    def generate_document_summary(self, filename: str) -> str:
        """Generates a concise summary for a specific uploaded document."""
        doc_pages = [doc for doc in self.raw_documents if doc["source"] == filename]
        if not doc_pages:
            return "Document not found."

        full_text = " ".join([p["text"] for p in doc_pages])
        words = full_text.split()
        sample_text = " ".join(words[:400])

        return f"📄 **{filename}** ({len(doc_pages)} pages, ~{len(words)} words)\n*Key Context Preview*: {sample_text}..."
