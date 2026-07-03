"""
chunker.py
Document Chunker — AlphaLens Literature Agent
Splits cleaned documents into overlapping text windows for embedding.
Success criteria: each chunk <= 512 tokens, overlap = 64 tokens,
chunk metadata retains source filename.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

CHUNK_SIZE = 512    # tokens
CHUNK_OVERLAP = 64  # tokens


def chunk_document(doc: Dict) -> List[Dict]:
    """
    Chunk a parsed document into overlapping text windows.
    Returns List of {text, chunk_id, doc_id, metadata}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(doc["full_text"])
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "chunk_id": f"{doc['filename']}_chunk_{i:04d}",
            "doc_id": doc["filename"],
            "text": chunk_text,
            "metadata": {**doc.get("metadata", {}), "chunk_index": i},
        })
    return result


def chunk_all_documents(docs: List[Dict]) -> List[Dict]:
    """Process all documents and return flat list of chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
