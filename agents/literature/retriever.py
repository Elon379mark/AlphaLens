from typing import List, Dict
from sentence_transformers import CrossEncoder

from agents.literature.vector_store import query_collection

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton so we don't reload the reranker on every call
_reranker_instance = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it across calls."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


def retrieve(collection, query_embedding: List[float], n_results: int = 20) -> List[Dict]:
    """
    Stage 1: Retrieve top-N candidates from ChromaDB (fast, approximate).
    Returns list of {chunk_id, text, score, metadata}.
    """
    results = query_collection(collection, query_embedding, n_results=n_results)
    retrieved = []
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": doc,
            "score": 1 - dist,  # cosine distance -> similarity
            "metadata": meta,
        })
    return retrieved


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Stage 2: Cross-encoder reranking for precision.
    Returns top_k reranked chunks, sorted best-first.
    """
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    from agents.literature.vector_store import get_chroma_client, get_or_create_collection
    from agents.literature.embedder import load_embedding_model, embed_query

    print("Connecting to ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    print(f"Collection has {collection.count()} vectors.")

    print("Loading embedding model...")
    embed_model = load_embedding_model()

    test_query = "momentum signal predictive power"
    query_vec = embed_query(test_query, embed_model)

    print(f"\nStage 1: Dense retrieval for '{test_query}'...")
    candidates = retrieve(collection, query_vec, n_results=20)
    print(f"Retrieved {len(candidates)} candidates.")
    print("\nTop 3 by dense score (before reranking):")
    for i, c in enumerate(sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]):
        print(f"{i+1}. {c['chunk_id']} | dense score: {c['score']:.4f}")

    print("\nStage 2: Reranking with cross-encoder (downloads ~120MB on first run)...")
    reranked = rerank(test_query, candidates, top_k=5)

    print("\nTop 5 after reranking:")
    for i, c in enumerate(reranked):
        print(f"{i+1}. {c['chunk_id']} | rerank score: {c['rerank_score']:.4f} | dense score: {c['score']:.4f}")
        print(f"   preview: {c['text'][:120]}...")