"""
extractor.py
JSON Fact Extractor — AlphaLens Literature Agent
Uses Groq (free) with llama-3.3-70b for structured fact extraction.
Logs failures but continues processing.
"""

import json
import os
import logging
from typing import List, Dict

from groq import Groq
from .prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def extract_facts_from_chunks(
    chunks: List[Dict],
    model: str = "llama-3.3-70b-versatile",
) -> List[Dict]:
    """
    For each retrieved chunk, call Groq LLM and extract structured facts.
    Returns flat list of fact dicts.
    Skips chunks where LLM returns invalid JSON (logs warning).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY is not set. Fact extraction will be skipped.")
        return []

    client = Groq(api_key=api_key)
    all_facts = []

    for chunk in chunks:
        prompt = build_extraction_prompt(chunk["text"])
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user",   "content": prompt["user"]},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content.strip()
            # Strip markdown fences if model accidentally includes them
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            facts = json.loads(raw_text)
            if isinstance(facts, list):
                for fact in facts:
                    fact["source_chunk_id"] = chunk["chunk_id"]
                all_facts.extend(facts)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Extraction failed for {chunk['chunk_id']}: {e}")

    return all_facts


def save_facts(
    facts: List[Dict],
    path: str = "outputs/literature_facts.json",
) -> None:
    """Persist extracted facts to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)
    logger.info(f"Saved {len(facts)} facts -> {path}")
