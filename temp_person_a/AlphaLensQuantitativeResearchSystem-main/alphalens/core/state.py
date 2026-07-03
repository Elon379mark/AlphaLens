"""
core/state.py
Shared LangGraph State Schema — AlphaLens
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, List, Dict, Optional, Any
import pandas as pd


class AlphaLensState(TypedDict):
    # === Literature Agent ===
    literature_facts: List[Dict]       # Extracted JSON facts per paper
    relevant_chunks: List[str]         # Retrieved text chunks
    signal_hypotheses: List[str]       # Hypotheses from literature

    # === Signal Generation ===
    raw_features: Any                  # pd.DataFrame — all 312 raw features
    validated_features: Any            # pd.DataFrame — post-IC-filter features
    ic_scores: Dict[str, float]        # IC per feature
    icir_scores: Dict[str, float]      # ICIR per feature
    ranked_signals: List[str]          # Signal names ranked by ICIR

    # === Deep Learning ===
    tft_predictions: Any               # pd.DataFrame — TFT forward returns
    nbeats_predictions: Any            # pd.DataFrame — N-BEATS forward returns
    patchtst_predictions: Any          # pd.DataFrame — PatchTST forward returns
    ensemble_predictions: Any          # pd.DataFrame — weighted ensemble

    # === GAT ===
    gat_embeddings: Dict[str, Any]     # Node embeddings per ticker
    graph_edges: List[tuple]           # (src, dst, weight) edges

    # === Causal ===
    dag_structure: Dict                # Discovered DAG edges
    ate_estimates: Dict[str, float]    # ATE per signal
    causal_signals: List[str]          # Causally validated signals

    # === Portfolio ===
    expected_returns: Any              # pd.Series — mu vector
    covariance_matrix: Any             # pd.DataFrame — Sigma matrix
    cvar_weights: Any                  # pd.Series — CVaR-optimal weights
    bl_weights: Any                    # pd.Series — Black-Litterman weights
    final_weights: Any                 # pd.Series — final portfolio weights
    portfolio_metrics: Dict            # Sharpe, CVaR, turnover etc.

    # === Metadata ===
    run_id: str
    universe: List[str]                # Ticker list
    as_of_date: str                    # ISO date string
    errors: List[str]                  # Agent error log
    logs: List[str]                    # Execution log
