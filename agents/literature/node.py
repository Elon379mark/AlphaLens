import os
from typing import Dict

from agents.literature.ingestion import load_pdf_directory
from agents.literature.parser import clean_text, extract_paper_metadata, remove_references_section
from agents.literature.chunker import chunk_all_documents
from agents.literature.embedder import load_embedding_model, embed_chunks, embed_query
from agents.literature.vector_store import get_chroma_client, get_or_create_collection, upsert_chunks
from agents.literature.retriever import retrieve, rerank
from agents.literature.extractor import extract_facts_from_chunks, save_facts

from core.state import AlphaLensState

PDF_DIR = os.getenv("PDF_DIR", "data/raw/pdfs")


def literature_agent_node(state: AlphaLensState) -> AlphaLensState:
    """
    LangGraph node: full literature RAG pipeline.
    Ingests PDFs, chunks, embeds, stores, retrieves per hypothesis,
    reranks, and extracts structured facts.
    """
    logs = list(state.get("logs", []))
    errors = list(state.get("errors", []))

    # --- Ingestion + Cleaning ---
    logs.append("literature_agent: starting PDF ingestion")
    docs = load_pdf_directory(PDF_DIR)
    for d in docs:
        d["full_text"] = remove_references_section(clean_text(d["full_text"]))
        d["metadata"] = extract_paper_metadata(d["full_text"])
    logs.append(f"literature_agent: ingested {len(docs)} documents")

    # --- Chunking + Embedding ---
    chunks = chunk_all_documents(docs)
    model = load_embedding_model()
    chunks = embed_chunks(chunks, model)
    logs.append(f"literature_agent: created {len(chunks)} chunks")

    # --- Store ---
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    upsert_chunks(collection, chunks)
    logs.append(f"literature_agent: upserted chunks, collection size = {collection.count()}")

    # --- Retrieve per hypothesis ---
    hypotheses = state.get("signal_hypotheses") or ["momentum", "value", "quality"]
    all_chunks = []
    for hyp in hypotheses:
        try:
            q_emb = embed_query(hyp, model)
            cands = retrieve(collection, q_emb, n_results=20)
            reranked = rerank(hyp, cands, top_k=5)
            all_chunks.extend(reranked)
        except Exception as e:
            errors.append(f"literature_agent: retrieval failed for hypothesis '{hyp}': {e}")

    # --- Deduplicate ---
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_chunks.append(c)
    logs.append(f"literature_agent: {len(unique_chunks)} unique chunks after dedup")

    # --- Extract facts ---
    facts = extract_facts_from_chunks(unique_chunks)
    save_facts(facts)
    logs.append(f"literature_agent: extracted {len(facts)} facts")

    return {
        **state,
        "literature_facts": facts,
        "relevant_chunks": [c["text"] for c in unique_chunks],
        "signal_hypotheses": hypotheses,
        "logs": logs,
        "errors": errors,
    }


if __name__ == "__main__":
    test_state: AlphaLensState = {
        "run_id": "test-run-001",
        "universe": [],
        "as_of_date": "2026-07-11",
        "signal_hypotheses": ["momentum", "value"],
        "errors": [],
        "logs": [],
    }

    print("Running full literature_agent_node end-to-end...\n")
    result = literature_agent_node(test_state)

    print("\n=== RESULT SUMMARY ===")
    print(f"Facts extracted: {len(result['literature_facts'])}")
    print(f"Relevant chunks: {len(result['relevant_chunks'])}")
    print(f"Hypotheses used: {result['signal_hypotheses']}")
    print(f"\nLogs:")
    for line in result["logs"]:
        print(f"  - {line}")
    if result["errors"]:
        print(f"\nErrors:")
        for line in result["errors"]:
            print(f"  - {line}")

    assert len(result["literature_facts"]) >= 0, "literature_facts key missing or malformed"
    assert len(result["relevant_chunks"]) > 0, "No chunks retrieved — check pipeline"
    print("\nPASS: literature_agent_node completed successfully.")