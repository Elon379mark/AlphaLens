"""
embedder.py
Embedding Pipeline — AlphaLens Literature Agent
Uses BAAI/bge-large-en-v1.5 (1024-dim) for strong finance performance.
Processes chunks in batches for memory efficiency.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024-dim, strong finance performance


def load_embedding_model() -> SentenceTransformer:
    """Load and return the sentence transformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Add 'embedding' (List[float]) to each chunk dict.
    Processes in batches for memory efficiency.
    Returns chunks with embedding field populated.
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def embed_query(query: str, model: SentenceTransformer) -> List[float]:
    """Embed a single query string for retrieval."""
    return model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()
