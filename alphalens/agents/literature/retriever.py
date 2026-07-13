"""
retriever.py
Retrieval & Reranking — AlphaLens Literature Agent
Two-stage retrieval:
  1. Dense retrieval via ChromaDB cosine similarity (top-N candidates)
  2. Cross-encoder reranking for precision (top-K final results)
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder
from .vector_store import query_collection

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def retrieve(
    collection,
    query_embedding: List[float],
    n_results: int = 20,
) -> List[Dict]:
    """
    Retrieve top-N candidates from vector store.
    Returns list of {chunk_id, text, score, metadata}.
    """
    results = query_collection(collection, query_embedding, n_results)
    retrieved = []
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": doc,
            "score": 1 - dist,  # cosine similarity
            "metadata": meta,
        })
    return retrieved


import torch

_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker_instance = CrossEncoder(RERANKER_MODEL, device=device)
    return _reranker_instance

def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Cross-encoder reranking for precision.
    Returns top_k reranked chunks.
    """
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
