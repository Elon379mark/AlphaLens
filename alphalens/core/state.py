"""
core/state.py
Shared LangGraph State Schema — AlphaLens
All agents read from and write to this TypedDict.
This is the consolidated, canonical state schema for the entire platform.
"""

from typing import TypedDict, List, Dict, Optional, Any, Tuple
from typing_extensions import NotRequired
import pandas as pd


class AlphaLensState(TypedDict):
    # === Workflow Metadata & Routing (from GraphState) ===
    run_id: str                          # Unique identifier for this pipeline run
    iteration: NotRequired[int]          # Refinement loop counter (max 3 before hard stop)
    created_at: NotRequired[str]         # ISO timestamp of pipeline start
    status: NotRequired[str]             # Current pipeline status
    current_node: NotRequired[str]       # For tracking execution in tests
    routing_decision: NotRequired[str]   # Last routing outcome for audit log
    error_message: NotRequired[str]      # Captures any node-level errors
    universe: NotRequired[List[str]]      # Ticker list
    as_of_date: NotRequired[str]          # ISO date string
    errors: NotRequired[List[str]]        # Agent error log
    logs: NotRequired[List[str]]          # Execution log

    # === Literature Agent Output (from both schemas) ===
    hypothesis_id: NotRequired[str]                   # Unique hypothesis key (UUID string)
    predictor_variable: NotRequired[str]             # e.g. "momentum_12_1", "earnings_surprise"
    target_asset_class: NotRequired[str]             # e.g. "US_EQUITY", "CRYPTO"
    predicted_direction: NotRequired[str]            # Enum: "positive" | "negative"
    hypothesis_confidence: NotRequired[float]        # Float [0.00, 1.00]
    raw_literature_context: NotRequired[str]         # Raw extracted text from RAG pipeline
    literature_facts: NotRequired[List[Dict]]         # Extracted JSON facts per paper
    relevant_chunks: NotRequired[List[str]]           # Retrieved text chunks
    signal_hypotheses: NotRequired[List[str]]         # Hypotheses from literature

    # === Signal Agent Output (from both schemas) ===
    feature_matrix_path: NotRequired[str]             # Path/reference to computed feature matrix
    information_coefficient: NotRequired[float]       # Spearman rank IC score
    information_ratio: NotRequired[float]             # ICIR = IC / std(IC)
    signal_computed_at: NotRequired[str]              # ISO timestamp
    raw_features: NotRequired[Any]                    # pd.DataFrame — all 312 raw features
    validated_features: NotRequired[Any]              # pd.DataFrame — post-IC-filter features
    ic_scores: NotRequired[Dict[str, float]]          # IC per feature
    icir_scores: NotRequired[Dict[str, float]]        # ICIR per feature
    ranked_signals: NotRequired[List[str]]            # Signal names ranked by ICIR

    # === Deep Learning (from AlphaLensState) ===
    tft_predictions: NotRequired[Any]                 # pd.DataFrame — TFT forward returns
    nbeats_predictions: NotRequired[Any]              # pd.DataFrame — N-BEATS forward returns
    patchtst_predictions: NotRequired[Any]            # pd.DataFrame — PatchTST forward returns
    ensemble_predictions: NotRequired[Any]            # pd.DataFrame — weighted ensemble

    # === GNN & Graphs (from AlphaLensState) ===
    gat_embeddings: NotRequired[Dict[str, Any]]       # Node embeddings per ticker
    graph_edges: NotRequired[List[Tuple[str, str, float]]]  # (src, dst, weight) edges

    # === Causal Validation Agent Output (from both schemas) ===
    p_value: NotRequired[float]                       # Statistical significance (Contract 2 threshold: < 0.05)
    ate_magnitude: NotRequired[float]                 # Average Treatment Effect from DML estimator
    dag_path: NotRequired[str]                        # Path to serialized DAG structure
    sharpe_ratio: NotRequired[float]                  # Sharpe ratio of signal (Contract 2 threshold: >= 1.0)
    causal_validated_at: NotRequired[str]             # ISO timestamp
    dag_structure: NotRequired[Dict]                  # Discovered DAG edges
    ate_estimates: NotRequired[Dict[str, float]]      # ATE per signal
    causal_signals: NotRequired[List[str]]            # Causally validated signals

    # === Portfolio Agent Output (from both schemas) ===
    portfolio_weights: NotRequired[Dict[str, float]]  # Asset -> weight mapping
    expected_returns: NotRequired[Any]                # pd.Series — mu vector
    expected_return: NotRequired[float]               # Black-Litterman expected return
    covariance_matrix: NotRequired[Any]               # pd.DataFrame — Sigma matrix
    cvar_weights: NotRequired[Any]                    # pd.Series — CVaR-optimal weights
    bl_weights: NotRequired[Any]                      # pd.Series — Black-Litterman weights
    final_weights: NotRequired[Any]                   # pd.Series — final portfolio weights
    portfolio_metrics: NotRequired[Dict]              # Sharpe, CVaR, turnover etc.
    portfolio_cvx_at: NotRequired[str]                # ISO timestamp

    # === Rejection / Refinement Output (from GraphState) ===
    rejection_reason: NotRequired[str]                # Human-readable rejection explanation
    refinement_suggestions: NotRequired[List[str]]    # Hints passed back to literature agent

    # === Test & Compatibility fields (from GraphState) ===
    hypothesis: NotRequired[Any]                      # For tracking full hypothesis schemas in tests
    half_life_days: NotRequired[float]                # Mock signal duration in tests
