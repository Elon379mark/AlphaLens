"""
parser.py
Text Parser — AlphaLens Literature Agent
Cleans raw PDF text, extracts paper metadata, and strips reference sections.
"""

import re
from typing import Dict


def clean_text(text: str) -> str:
    """
    Remove headers, footers, figure captions, references section.
    Normalize whitespace.
    """
    text = re.sub(r'\f', '\n', text)           # form feeds
    text = re.sub(r'\n{3,}', '\n\n', text)     # excessive newlines
    text = re.sub(r'- *\n', '', text)           # hyphenation
    text = re.sub(r'[^\S\n]+', ' ', text)       # horizontal whitespace
    return text.strip()


def extract_paper_metadata(text: str) -> Dict:
    """
    Heuristically extract title, year, abstract from raw text.
    Returns {title, year, abstract}.
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
    """Strip everything after REFERENCES heading."""
    match = re.search(r'\nREFERENCES\b', text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    return text
