"""
ingestion.py
PDF Ingestion — AlphaLens Literature Agent
Loads PDFs from a directory. Uses pymupdf (fitz) for extraction,
falls back to pypdf if needed.
"""

import os
from pathlib import Path
from typing import List, Dict

import pymupdf  # fitz
from pypdf import PdfReader


def load_pdf_directory(pdf_dir: str) -> List[Dict]:
    """
    Load all PDFs from a directory.
    Returns list of {filename, full_text, pages, metadata}.
    """
    docs = []
    pdf_dir = Path(pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc = load_single_pdf(str(pdf_path))
        docs.append(doc)
    return docs


def load_single_pdf(path: str) -> Dict:
    """
    Load one PDF. Uses pymupdf for text; falls back to pypdf.
    Returns {filename, full_text, pages: List[str], num_pages, path, source}.
    """
    try:
        doc = pymupdf.open(path)
        pages = [page.get_text("text") for page in doc]
        full_text = "\n\n".join(pages)
        return {
            "filename": Path(path).name,
            "path": path,
            "full_text": full_text,
            "pages": pages,
            "num_pages": len(pages),
            "source": "pymupdf",
        }
    except Exception:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return {
            "filename": Path(path).name,
            "path": path,
            "full_text": "\n\n".join(pages),
            "pages": pages,
            "num_pages": len(reader.pages),
            "source": "pypdf",
        }


def extract_metadata(doc: Dict) -> Dict:
    """
    Extract basic metadata from a loaded document dict.
    Returns {filename, num_pages, source, path}.
    """
    return {
        "filename": doc.get("filename", ""),
        "num_pages": doc.get("num_pages", 0),
        "source": doc.get("source", ""),
        "path": doc.get("path", ""),
    }
