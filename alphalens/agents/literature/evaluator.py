"""
evaluator.py
RAG Evaluation — AlphaLens Literature Agent
Computes P@K, R@K, and MRR against a labelled query set.
Target: MRR > 0.50 on held-out query set.
"""

from typing import List, Dict, Callable


def precision_at_k(
    retrieved: List[Dict],
    relevant_ids: List[str],
    k: int = 5,
) -> float:
    """P@K: fraction of top-K results that are relevant."""
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r["chunk_id"] in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved: List[Dict],
    relevant_ids: List[str],
    k: int = 5,
) -> float:
    """R@K: fraction of relevant results found in top-K."""
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r["chunk_id"] in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved: List[Dict], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank."""
    for rank, r in enumerate(retrieved, 1):
        if r["chunk_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_rag_evaluation(
    retriever_fn: Callable,
    test_queries: List[Dict],
) -> Dict:
    """
    Evaluate RAG retrieval on a labelled query set.

    test_queries: [{query, relevant_chunk_ids}]
    Returns {avg_precision_5, avg_recall_5, avg_mrr}.
    """
    p5_scores, r5_scores, mrr_scores = [], [], []
    for q in test_queries:
        results = retriever_fn(q["query"])
        p5_scores.append(precision_at_k(results, q["relevant_chunk_ids"]))
        r5_scores.append(recall_at_k(results, q["relevant_chunk_ids"]))
        mrr_scores.append(mrr(results, q["relevant_chunk_ids"]))
    return {
        "avg_precision_5": sum(p5_scores) / len(p5_scores),
        "avg_recall_5": sum(r5_scores) / len(r5_scores),
        "avg_mrr": sum(mrr_scores) / len(mrr_scores),
    }
