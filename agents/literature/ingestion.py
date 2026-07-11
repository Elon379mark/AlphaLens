import os
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
import pymupdf  # fitz


def load_pdf_directory(pdf_dir: str) -> List[Dict]:
    """
    Load all PDFs from a directory.
    Returns list of {filename, path, full_text, pages, num_pages, source}
    """
    docs = []
    pdf_dir_path = Path(pdf_dir)

    if not pdf_dir_path.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    for pdf_path in pdf_files:
        doc = load_single_pdf(str(pdf_path))
        docs.append(doc)

    return docs


def load_single_pdf(path: str) -> Dict:
    """
    Load one PDF. Uses pymupdf for text; falls back to pypdf.
    Returns {filename, path, full_text, pages, num_pages, source}
    """
    try:
        doc = pymupdf.open(path)
        pages = [page.get_text("text") for page in doc]
        full_text = "\n\n".join(pages)
        doc.close()
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
    Extract basic metadata: filename, page count, character count.
    Returns {filename, num_pages, char_count}
    """
    return {
        "filename": doc["filename"],
        "num_pages": doc["num_pages"],
        "char_count": len(doc["full_text"]),
    }


if __name__ == "__main__":
    # Quick manual test when running this file directly
    PDF_DIR = "data/raw/pdfs"
    documents = load_pdf_directory(PDF_DIR)
    for d in documents:
        meta = extract_metadata(d)
        print(f"Loaded: {meta['filename']} | Pages: {meta['num_pages']} | Chars: {meta['char_count']} | Extractor: {d['source']}")