from typing import TypedDict, List, Dict, Optional, Any


import pandas as pd


class AlphaLensState(TypedDict, total=False):
    # === Literature Agent ===
    literature_facts: List[Dict]       # Extracted JSON facts per paper
    relevant_chunks: List[str]         # Retrieved text chunks
    signal_hypotheses: List[str]       # Hypotheses / query topics used for retrieval

    # === Signal Generation ===
    raw_features: pd.DataFrame         # All 312 raw features
    validated_features: pd.DataFrame   # Post-IC/ICIR-filter features
    ic_scores: Dict[str, float]        # IC per feature
    icir_scores: Dict[str, float]      # ICIR per feature
    ranked_signals: List[str]          # Signal names ranked by ICIR
    # === Causal ===
    dag_structure: Dict
    ate_estimates: Dict[str, dict]
    causal_signals: List[str]

    # === Metadata ===
    run_id: str
    universe: List[str]                # Ticker list (used by later agents)
    as_of_date: str                    # ISO date string
    errors: List[str]                  # Agent error log
    logs: List[str]                    # Execution log