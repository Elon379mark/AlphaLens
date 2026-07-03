"""
node.py
LangGraph Node — AlphaLens Literature Agent
Full RAG pipeline wired as a single LangGraph node.
Reads PDF_DIR from env, runs ingestion → chunking → embedding →
vector store → retrieval → reranking → LLM extraction,
then returns updated AlphaLensState.
"""

import os
from typing import Any, Dict

from .ingestion import load_pdf_directory
from .parser import clean_text, extract_paper_metadata, remove_references_section
from .chunker import chunk_all_documents
from .embedder import load_embedding_model, embed_chunks, embed_query
from .vector_store import get_chroma_client, get_or_create_collection, upsert_chunks
from .retriever import retrieve, rerank
from .extractor import extract_facts_from_chunks, save_facts

PDF_DIR = os.getenv("PDF_DIR", "data/raw/pdfs")


def literature_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: full literature RAG pipeline.

    Steps:
      1. Ingest PDFs from PDF_DIR
      2. Clean text, extract metadata, remove references
      3. Chunk documents
      4. Embed chunks
      5. Upsert into ChromaDB
      6. For each hypothesis: retrieve → rerank
      7. Deduplicate chunks
      8. Extract structured JSON facts via LLM
      9. Return updated state
    """
    # --- 1. Ingestion ---
    docs = load_pdf_directory(PDF_DIR)
    for d in docs:
        d["full_text"] = remove_references_section(clean_text(d["full_text"]))
        d["metadata"] = extract_paper_metadata(d["full_text"])

    # --- 2. Chunking + Embedding ---
    chunks = chunk_all_documents(docs)
    model = load_embedding_model()
    chunks = embed_chunks(chunks, model)

    # --- 3. Store ---
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    upsert_chunks(collection, chunks)

    # --- 4. Retrieve for each hypothesis ---
    hypotheses = state.get("signal_hypotheses", ["momentum", "value", "quality"])
    all_chunks = []
    for hyp in hypotheses:
        q_emb = embed_query(hyp, model)
        cands = retrieve(collection, q_emb, n_results=20)
        reranked = rerank(hyp, cands, top_k=5)
        all_chunks.extend(reranked)

    # --- 5. Deduplicate ---
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_chunks.append(c)

    # --- 6. Extract facts ---
    facts = extract_facts_from_chunks(unique_chunks)
    save_facts(facts)

    return {
        **state,
        "literature_facts": facts,
        "relevant_chunks": [c["text"] for c in unique_chunks],
        "signal_hypotheses": hypotheses,
    }
