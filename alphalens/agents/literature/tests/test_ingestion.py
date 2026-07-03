"""
test_ingestion.py
Tests for PDF ingestion pipeline.
Asserts: >= 1 page extracted, no empty strings, metadata keys present.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.literature.ingestion import (
    load_pdf_directory,
    load_single_pdf,
    extract_metadata,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_PDF_DIR = "data/raw/pdfs"


def _make_mock_doc(filename="test_paper.pdf", num_pages=3):
    return {
        "filename": filename,
        "path": f"data/raw/pdfs/{filename}",
        "full_text": "Abstract\nThis paper introduces momentum signals.\n\nIntroduction\nMomentum is well documented.",
        "pages": ["Page 1 text", "Page 2 text", "Page 3 text"][:num_pages],
        "num_pages": num_pages,
        "source": "pymupdf",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadSinglePdf:
    def test_returns_required_keys(self, tmp_path):
        """Returned dict must contain all required keys."""
        # Create a minimal valid PDF using pypdf (write-only) or mock
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Test momentum signal research paper.", ln=True)
        pdf_path = str(tmp_path / "test.pdf")
        pdf.output(pdf_path)

        doc = load_single_pdf(pdf_path)
        for key in ["filename", "path", "full_text", "pages", "num_pages", "source"]:
            assert key in doc, f"Missing key: {key}"

    def test_num_pages_positive(self, tmp_path):
        from fpdf import FPDF
        pdf = FPDF()
        for _ in range(2):
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Content.", ln=True)
        pdf_path = str(tmp_path / "multi.pdf")
        pdf.output(pdf_path)

        doc = load_single_pdf(pdf_path)
        assert doc["num_pages"] >= 1

    def test_full_text_not_empty(self, tmp_path):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Value factor paper.", ln=True)
        pdf_path = str(tmp_path / "value.pdf")
        pdf.output(pdf_path)

        doc = load_single_pdf(pdf_path)
        assert len(doc["full_text"].strip()) > 0

    def test_pages_list_not_empty(self, tmp_path):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Quality factor.", ln=True)
        pdf_path = str(tmp_path / "quality.pdf")
        pdf.output(pdf_path)

        doc = load_single_pdf(pdf_path)
        assert isinstance(doc["pages"], list)
        assert len(doc["pages"]) >= 1


class TestLoadPdfDirectory:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        result = load_pdf_directory(str(tmp_path))
        assert result == []

    def test_loads_multiple_pdfs(self, tmp_path):
        from fpdf import FPDF
        for name in ["paper_a.pdf", "paper_b.pdf"]:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Content of {name}.", ln=True)
            pdf.output(str(tmp_path / name))

        docs = load_pdf_directory(str(tmp_path))
        assert len(docs) == 2

    def test_results_sorted_by_filename(self, tmp_path):
        from fpdf import FPDF
        for name in ["z_paper.pdf", "a_paper.pdf"]:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Content.", ln=True)
            pdf.output(str(tmp_path / name))

        docs = load_pdf_directory(str(tmp_path))
        filenames = [d["filename"] for d in docs]
        assert filenames == sorted(filenames)


class TestExtractMetadata:
    def test_metadata_keys(self):
        doc = _make_mock_doc()
        meta = extract_metadata(doc)
        for key in ["filename", "num_pages", "source", "path"]:
            assert key in meta

    def test_filename_preserved(self):
        doc = _make_mock_doc(filename="momentum_paper.pdf")
        meta = extract_metadata(doc)
        assert meta["filename"] == "momentum_paper.pdf"

    def test_num_pages_correct(self):
        doc = _make_mock_doc(num_pages=5)
        meta = extract_metadata(doc)
        assert meta["num_pages"] == 5
