"""
test_chunker.py
Tests for document chunking.
Success criteria:
  - Each chunk <= 512 tokens
  - Chunk metadata retains source filename
  - chunk_id is unique within a document
"""

import pytest
from agents.literature.chunker import (
    chunk_document,
    chunk_all_documents,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(filename="paper.pdf", word_count=2000):
    """Generate a synthetic doc dict with word_count words."""
    text = " ".join(["word"] * word_count)
    return {
        "filename": filename,
        "full_text": text,
        "metadata": {"title": "Test Paper", "year": 2024},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChunkDocument:
    def test_returns_list_of_dicts(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_has_required_keys(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        assert len(chunks) > 0
        for c in chunks:
            for key in ["chunk_id", "doc_id", "text", "metadata"]:
                assert key in c, f"Missing key: {key}"

    def test_chunk_ids_are_unique(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_text_within_size_limit(self):
        """Each chunk text should be within ~512 tokens (approximated by chars/4)."""
        doc = _make_doc(word_count=5000)
        chunks = chunk_document(doc)
        for c in chunks:
            # Rough token estimate: chars / 4
            approx_tokens = len(c["text"]) / 4
            assert approx_tokens <= CHUNK_SIZE * 1.2, (
                f"Chunk too large: ~{approx_tokens:.0f} tokens"
            )

    def test_doc_id_matches_filename(self):
        doc = _make_doc(filename="momentum_paper.pdf")
        chunks = chunk_document(doc)
        assert all(c["doc_id"] == "momentum_paper.pdf" for c in chunks)

    def test_metadata_carries_chunk_index(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            assert "chunk_index" in c["metadata"]

    def test_chunk_id_contains_filename(self):
        doc = _make_doc(filename="value_paper.pdf")
        chunks = chunk_document(doc)
        assert all("value_paper.pdf" in c["chunk_id"] for c in chunks)

    def test_short_doc_produces_at_least_one_chunk(self):
        doc = {
            "filename": "short.pdf",
            "full_text": "This is a short abstract about momentum investing.",
            "metadata": {},
        }
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_metadata_from_doc_is_preserved(self):
        doc = _make_doc()
        doc["metadata"]["title"] = "AlphaLens Paper"
        chunks = chunk_document(doc)
        assert all(c["metadata"].get("title") == "AlphaLens Paper" for c in chunks)


class TestChunkAllDocuments:
    def test_flat_list_returned(self):
        docs = [_make_doc(f"paper_{i}.pdf") for i in range(3)]
        all_chunks = chunk_all_documents(docs)
        assert isinstance(all_chunks, list)
        assert len(all_chunks) > 0

    def test_chunk_count_grows_with_docs(self):
        docs_1 = [_make_doc("a.pdf")]
        docs_2 = [_make_doc("a.pdf"), _make_doc("b.pdf")]
        assert len(chunk_all_documents(docs_2)) > len(chunk_all_documents(docs_1))

    def test_empty_docs_returns_empty(self):
        assert chunk_all_documents([]) == []

    def test_no_duplicate_chunk_ids_across_docs(self):
        docs = [_make_doc(f"paper_{i}.pdf") for i in range(3)]
        chunks = chunk_all_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
