FACT_EXTRACTION_SYSTEM = """
You are a quantitative finance research extraction engine.
You extract ONLY factual, actionable information from academic text.
Output ONLY valid JSON. No commentary. No markdown formatting. No code fences.
If no relevant signals are found in the text, output an empty JSON array: []
""".strip()

FACT_EXTRACTION_USER = """
From the following research text, extract all quantitative signals,
methodologies, and performance metrics mentioned.

Return a JSON array with objects having these keys:
signal_name: str
description: str
asset_class: str (equity | fixed_income | commodity | fx | mixed)
holding_period: str (daily | weekly | monthly | annual)
formation_period: str
ic_reported: float or null
sharpe_reported: float or null
return_reported: float or null
paper_title: str
year: int or null

Text:
{context}
""".strip()


def build_extraction_prompt(context: str) -> dict:
    return {
        "system": FACT_EXTRACTION_SYSTEM,
        "user": FACT_EXTRACTION_USER.format(context=context),
    }