import re
from typing import Dict


def clean_text(text: str) -> str:
    """
    Remove form feeds, excessive newlines, hyphenation breaks,
    and normalize whitespace.
    """
    text = re.sub(r'\f', '\n', text)                # form feeds -> newline
    text = re.sub(r'\n{3,}', '\n\n', text)           # collapse excessive newlines
    text = re.sub(r'-\s*\n', '', text)               # rejoin hyphenated line breaks
    text = re.sub(r'[^\S\n]+', ' ', text)            # collapse horizontal whitespace
    return text.strip()


def extract_paper_metadata(text: str) -> Dict:
    """
    Heuristically extract title, year, and abstract from raw text.
    Returns {title, year, abstract}
    """
    lines = text.split('\n')
    title = lines[0].strip() if lines else "Unknown"

    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    year = int(year_match.group()) if year_match else None

    abstract_match = re.search(
        r'(?i)abstract[:\s]+(.*?)(?=\n\n|\nintroduction|\nkeywords)',
        text, re.DOTALL
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    return {"title": title, "year": year, "abstract": abstract}


def remove_references_section(text: str) -> str:
    """Strip everything after a References/Bibliography heading."""
    match = re.search(r'\n(REFERENCES|References|Bibliography)\b', text)
    if match:
        text = text[:match.start()]
    return text


if __name__ == "__main__":
    from agents.literature.ingestion import load_pdf_directory

    PDF_DIR = "data/raw/pdfs"
    docs = load_pdf_directory(PDF_DIR)

    for d in docs:
        raw_len = len(d["full_text"])
        cleaned = clean_text(d["full_text"])
        cleaned = remove_references_section(cleaned)
        meta = extract_paper_metadata(cleaned)

        print(f"--- {d['filename']} ---")
        print(f"Raw chars: {raw_len} | Cleaned chars: {len(cleaned)}")
        print(f"Detected title: {meta['title'][:80]}")
        print(f"Detected year: {meta['year']}")
        print(f"Abstract found: {'yes' if meta['abstract'] else 'no'}")
        print()