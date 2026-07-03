"""
vector_store.py
Vector Store Interface — AlphaLens Literature Agent
ChromaDB persistent vector store for embedded chunks.
Uses cosine similarity (hnsw:space = cosine).
Upserts in batches of 500 to avoid memory errors.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/vectors/chroma_db")
COLLECTION_NAME = "alphalens_literature"


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get or create the AlphaLens literature collection with cosine similarity."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: List[Dict]) -> None:
    """
    Upsert embedded chunks into ChromaDB.
    Chunks must have: chunk_id, text, embedding, metadata.
    Processes in batches of 500.
    """
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def query_collection(
    collection,
    query_embedding: List[float],
    n_results: int = 10,
) -> Dict:
    """
    Query ChromaDB for similar chunks.
    Returns {ids, documents, distances, metadatas}.
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )


def delete_collection(client: chromadb.PersistentClient) -> None:
    """Delete the collection (useful for testing / reset)."""
    client.delete_collection(name=COLLECTION_NAME)
