import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

os.makedirs("sample_documents", exist_ok=True)

def build_pdf_1():
    pdf_path = os.path.join("sample_documents", "Artificial_Intelligence_Overview.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Page 1
    story.append(Paragraph("Artificial Intelligence: Foundations & History", styles['Title']))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Section 1: Introduction to AI</b>", styles['Heading2']))
    story.append(Paragraph(
        "Artificial Intelligence (AI) is a field of computer science dedicated to creating systems "
        "capable of performing tasks that typically require human intelligence. These tasks include "
        "visual perception, speech recognition, decision-making, and natural language translation.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Section 2: Machine Learning Paradigms</b>", styles['Heading2']))
    story.append(Paragraph(
        "Machine Learning (ML) is a subset of AI where algorithms learn patterns from data rather than "
        "being explicitly programmed. The main paradigms are:<br/>"
        "1. <b>Supervised Learning</b>: Learning from labeled data (e.g., classification, regression).<br/>"
        "2. <b>Unsupervised Learning</b>: Finding hidden patterns in unlabeled data (e.g., clustering, PCA).<br/>"
        "3. <b>Reinforcement Learning</b>: Learning via rewards and penalties through interaction with an environment.",
        styles['Normal']
    ))

    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Deep Learning & Transformers", styles['Title']))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Section 3: Neural Networks & Deep Learning</b>", styles['Heading2']))
    story.append(Paragraph(
        "Deep Learning utilizes multi-layered artificial neural networks (ANNs) inspired by biological brain structures. "
        "Key architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks "
        "(RNNs) for sequential data.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Section 4: The Transformer Revolution</b>", styles['Heading2']))
    story.append(Paragraph(
        "Introduced by Vaswani et al. in 2017 ('Attention Is All You Need'), the Transformer architecture replaced RNNs "
        "using self-attention mechanisms. Transformers enable parallelized processing of long text sequences and "
        "form the foundation of Large Language Models (LLMs) like GPT-4, Gemini, and Llama 3.",
        styles['Normal']
    ))

    doc.build(story)
    print(f"Created {pdf_path}")

def build_pdf_2():
    pdf_path = os.path.join("sample_documents", "RAG_Architecture_Guide.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Page 1
    story.append(Paragraph("Retrieval-Augmented Generation (RAG) Guide", styles['Title']))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Chapter 1: What is RAG?</b>", styles['Heading2']))
    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances Large Language Models "
        "by fetching relevant factual chunks from external knowledge bases before generating an answer. "
        "This significantly reduces LLM hallucinations and allows models to access private or domain-specific documents.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Chapter 2: Text Processing & Chunking Strategies</b>", styles['Heading2']))
    story.append(Paragraph(
        "Document ingestion requires splitting large texts into manageable chunks.<br/>"
        "• <b>Chunk Size</b>: Typically between 500 to 1000 characters.<br/>"
        "• <b>Chunk Overlap</b>: Typically 100 to 200 characters to prevent loss of semantic context across boundaries.<br/>"
        "• <b>Recursive Chunking</b>: Splits text at paragraphs, sentences, and words recursively.",
        styles['Normal']
    ))

    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Vector Stores & Retrieval Accuracy", styles['Title']))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Chapter 3: Vector Databases (FAISS & Chroma)</b>", styles['Heading2']))
    story.append(Paragraph(
        "Embeddings convert text chunks into high-dimensional numerical vectors. Vector databases store these vectors "
        "and perform fast Approximate Nearest Neighbor (ANN) search using distance metrics like Cosine Similarity or Euclidean Distance.<br/>"
        "• <b>FAISS</b> (Facebook AI Similarity Search): Extremely fast GPU/CPU vector indexing library.<br/>"
        "• <b>ChromaDB</b>: Open-source developer-friendly vector database with persistence capabilities.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Chapter 4: Evaluation Metrics</b>", styles['Heading2']))
    story.append(Paragraph(
        "Key metrics for evaluating RAG systems include Context Precision, Context Recall, Answer Faithfulness, "
        "and Answer Relevance. Strict prompt templates ensure the LLM returns 'I couldn't find this information' "
        "when context is missing.",
        styles['Normal']
    ))

    doc.build(story)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    build_pdf_1()
    build_pdf_2()
