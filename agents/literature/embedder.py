from sentence_transformers import SentenceTransformer
from typing import List, Dict

MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024-dim, strong on finance/research text


def load_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model. First call downloads ~1.3GB and caches it
    in ~/.cache/huggingface — subsequent calls are fast.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: List[Dict], model: SentenceTransformer, batch_size: int = 32) -> List[Dict]:
    """
    Add an 'embedding' field (List[float], length 1024) to each chunk dict.
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
    return model.encode([query], normalize_embeddings=True)[0].tolist()


if __name__ == "__main__":
    from agents.literature.ingestion import load_pdf_directory
    from agents.literature.parser import clean_text, remove_references_section, extract_paper_metadata
    from agents.literature.chunker import chunk_all_documents

    PDF_DIR = "data/raw/pdfs"
    docs = load_pdf_directory(PDF_DIR)

    for d in docs:
        d["full_text"] = remove_references_section(clean_text(d["full_text"]))
        d["metadata"] = extract_paper_metadata(d["full_text"])

    all_chunks = chunk_all_documents(docs)
    print(f"Total chunks to embed: {len(all_chunks)}")

    print("Loading embedding model (first run downloads ~1.3GB, please wait)...")
    model = load_embedding_model()
    print("Model loaded.")

    embedded_chunks = embed_chunks(all_chunks, model)

    sample = embedded_chunks[0]
    print()
    print("--- Sample embedded chunk ---")
    print(f"chunk_id: {sample['chunk_id']}")
    print(f"embedding length: {len(sample['embedding'])}")
    print(f"first 5 values: {sample['embedding'][:5]}")

    # Sanity check: embed a test query and confirm same dimensionality
    query_vec = embed_query("momentum signal 12-1", model)
    print()
    print(f"Query embedding length: {len(query_vec)}")
    assert len(query_vec) == len(sample["embedding"]), "Dimension mismatch between chunk and query embeddings!"
    print("Dimension check passed: chunk and query embeddings match.")