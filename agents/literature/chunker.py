from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

CHUNK_SIZE = 512       # characters (approx. proxy for tokens at this stage)
CHUNK_OVERLAP = 64


def chunk_document(doc: Dict) -> List[Dict]:
    """
    Chunk a parsed document into overlapping text windows.
    Returns List of {chunk_id, doc_id, text, metadata}
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
    """Process all documents and return a flat list of chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    from agents.literature.ingestion import load_pdf_directory
    from agents.literature.parser import clean_text, remove_references_section, extract_paper_metadata

    PDF_DIR = "data/raw/pdfs"
    docs = load_pdf_directory(PDF_DIR)

    for d in docs:
        d["full_text"] = remove_references_section(clean_text(d["full_text"]))
        d["metadata"] = extract_paper_metadata(d["full_text"])

    all_chunks = chunk_all_documents(docs)

    print(f"Total documents: {len(docs)}")
    print(f"Total chunks: {len(all_chunks)}")
    print()
    print("--- Sample chunk ---")
    sample = all_chunks[0]
    print(f"chunk_id: {sample['chunk_id']}")
    print(f"doc_id: {sample['doc_id']}")
    print(f"length: {len(sample['text'])} chars")
    print(f"text preview: {sample['text'][:200]}...")