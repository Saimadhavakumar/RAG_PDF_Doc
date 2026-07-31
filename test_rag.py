import os
from rag_engine import RAGEngine

def test_pipeline():
    print("--- 1. Initializing RAG Engine ---")
    engine = RAGEngine()

    pdf_files = [
        ("Artificial_Intelligence_Overview.pdf", open(os.path.join("sample_documents", "Artificial_Intelligence_Overview.pdf"), "rb").read()),
        ("RAG_Architecture_Guide.pdf", open(os.path.join("sample_documents", "RAG_Architecture_Guide.pdf"), "rb").read())
    ]

    print("--- 2. Parsing PDFs ---")
    extracted_pages = engine.extract_text_from_pdfs(pdf_files)
    print(f"Extracted {len(extracted_pages)} total pages across documents.")
    assert len(extracted_pages) >= 4, "Page extraction failed."

    print("--- 3. Processing & Chunking ---")
    chunks = engine.process_and_chunk_text(chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")
    assert len(chunks) > 0, "Chunking failed."

    print("--- 4. Building Vector Store ---")
    msg = engine.build_vector_store(provider="huggingface")
    print(f"Vector Store Status: {msg}")

    print("--- 5. Semantic Search Retrieval ---")
    query = "What are the chunk size and overlap recommendations for RAG?"
    results = engine.retrieve_relevant_chunks(query, top_k=2)
    print(f"Query: '{query}'")
    for r in results:
        print(f"  -> [Source: {r['source']} | Page: {r['page']} | Score: {r['score']}] Snippet: {r['content'][:120]}...")

    assert len(results) > 0, "Retrieval returned no chunks."

    print("--- 6. Testing Groq Llama 3 Answer Generation ---")
    retrieved = engine.retrieve_relevant_chunks("What are the chunk size recommendations for RAG?", top_k=2)
    groq_key = os.environ.get("GROQ_API_KEY")
    res = engine.generate_answer(
        question="What are the chunk size recommendations for RAG?",
        retrieved_chunks=retrieved,
        llm_provider="groq",
        api_key=groq_key
    )
    print(f"Groq Llama 3 Response:\n{res['answer']}\n")
    assert len(res['answer']) > 10, "Groq answer generation failed."

    print("--- 7. Test Unanswerable Query (Fallback Prompt Test) ---")
    unanswerable = "What is the capital of Mars?"
    retrieved_unanswerable = engine.retrieve_relevant_chunks(unanswerable, top_k=2)
    res_unanswerable = engine.generate_answer(
        question=unanswerable,
        retrieved_chunks=retrieved_unanswerable,
        llm_provider="groq",
        api_key=groq_key
    )
    print(f"Groq Unanswerable Response:\n{res_unanswerable['answer']}\n")

    print("\n[OK] All RAG engine & Groq LLM tests PASSED successfully!")

if __name__ == "__main__":
    test_pipeline()
