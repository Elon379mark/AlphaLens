import chromadb
from typing import List, Dict

CHROMA_PATH = "data/vectors/chroma_db"
COLLECTION_NAME = "alphalens_literature"


def get_chroma_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client that saves to disk."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get the collection if it exists, or create it with cosine similarity."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: List[Dict]) -> None:
    """
    Upsert embedded chunks into ChromaDB.
    Each chunk must have: chunk_id, text, embedding, metadata.
    Batches of 500 to stay within ChromaDB limits.
    """
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=docs[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )


def query_collection(collection, query_embedding: List[float], n_results: int = 10) -> Dict:
    """
    Query ChromaDB for similar chunks.
    Returns {ids, documents, distances, metadatas}
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )


if __name__ == "__main__":
    from agents.literature.ingestion import load_pdf_directory
    from agents.literature.parser import clean_text, remove_references_section, extract_paper_metadata
    from agents.literature.chunker import chunk_all_documents
    from agents.literature.embedder import load_embedding_model, embed_chunks, embed_query

    PDF_DIR = "data/raw/pdfs"
    docs = load_pdf_directory(PDF_DIR)

    for d in docs:
        d["full_text"] = remove_references_section(clean_text(d["full_text"]))
        d["metadata"] = extract_paper_metadata(d["full_text"])

    all_chunks = chunk_all_documents(docs)
    print(f"Total chunks: {len(all_chunks)}")

    print("Loading embedding model...")
    model = load_embedding_model()
    embedded_chunks = embed_chunks(all_chunks, model)
    print("Embedding complete.")

    print("Connecting to ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    print("Upserting chunks into vector store...")
    upsert_chunks(collection, embedded_chunks)

    total_in_db = collection.count()
    print(f"Total vectors now in collection '{COLLECTION_NAME}': {total_in_db}")

    # Test retrieval with a real query
    test_query = "momentum signal"
    query_vec = embed_query(test_query, model)
    results = query_collection(collection, query_vec, n_results=3)

    print()
    print(f"--- Top 3 results for query: '{test_query}' ---")
    for i, (doc_id, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        print(f"{i+1}. {doc_id} | distance: {dist:.4f}")