"""
prompts.py
Prompt Templates — AlphaLens Literature Agent
System and user prompts for JSON fact extraction from research text.
LLM must return ONLY valid JSON — no commentary, no markdown fences.
"""

FACT_EXTRACTION_SYSTEM = """
You are a quantitative finance research extraction engine.
You extract ONLY factual, actionable information from academic text.
Output ONLY valid JSON. No commentary. No markdown formatting.
""".strip()

FACT_EXTRACTION_USER = """
From the following research text, extract all quantitative signals,
methodologies, and performance metrics.

Return a JSON array with objects having keys:
  signal_name      : str
  description      : str
  asset_class      : str (equity | fixed_income | commodity | fx | mixed)
  holding_period   : str (daily | weekly | monthly | annual)
  formation_period : str
  ic_reported      : float | null
  sharpe_reported  : float | null
  return_reported  : float | null
  paper_title      : str
  year             : int | null

Text:
{context}
""".strip()


def build_extraction_prompt(context: str) -> dict:
    """
    Build the system + user prompt dict for a given text chunk.
    Returns {system, user}.
    """
    return {
        "system": FACT_EXTRACTION_SYSTEM,
        "user": FACT_EXTRACTION_USER.format(context=context),
    }
