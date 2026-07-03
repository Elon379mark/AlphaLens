"""
AlphaLens — Literature Agent
Retrieval-Augmented Generation pipeline for quantitative finance research.
"""

from .node import literature_agent_node
from .ingestion import load_pdf_directory, load_single_pdf
from .parser import clean_text, extract_paper_metadata, remove_references_section
from .chunker import chunk_document, chunk_all_documents
from .embedder import load_embedding_model, embed_chunks, embed_query
from .vector_store import (
    get_chroma_client,
    get_or_create_collection,
    upsert_chunks,
    query_collection,
)
from .retriever import retrieve, rerank
from .extractor import extract_facts_from_chunks, save_facts
from .evaluator import precision_at_k, recall_at_k, mrr, run_rag_evaluation

__all__ = [
    "literature_agent_node",
    "load_pdf_directory",
    "load_single_pdf",
    "clean_text",
    "extract_paper_metadata",
    "remove_references_section",
    "chunk_document",
    "chunk_all_documents",
    "load_embedding_model",
    "embed_chunks",
    "embed_query",
    "get_chroma_client",
    "get_or_create_collection",
    "upsert_chunks",
    "query_collection",
    "retrieve",
    "rerank",
    "extract_facts_from_chunks",
    "save_facts",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "run_rag_evaluation",
]
