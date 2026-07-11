from typing import TypedDict, List, Dict, Optional, Any


class AlphaLensState(TypedDict, total=False):
    # === Literature Agent ===
    literature_facts: List[Dict]       # Extracted JSON facts per paper
    relevant_chunks: List[str]         # Retrieved text chunks
    signal_hypotheses: List[str]       # Hypotheses / query topics used for retrieval

    # === Metadata ===
    run_id: str
    universe: List[str]                # Ticker list (used by later agents)
    as_of_date: str                    # ISO date string
    errors: List[str]                  # Agent error log
    logs: List[str]                    # Execution log