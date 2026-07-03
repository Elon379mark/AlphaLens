"""
node.py
LangGraph Node — AlphaLens Signal Generation Agent

Wires the full feature factory pipeline into a single LangGraph node function:
  1. Load OHLCV + fundamentals
  2. Compute all 312 features
  3. Compute forward returns
  4. Compute IC / ICIR per feature
  5. Validate features (NaN, IC, ICIR thresholds)
  6. Rank validated signals
  7. Persist outputs and update shared state
"""

import os
import json
import pandas as pd

from .data_loader import load_ohlcv, load_fundamentals
from .features import compute_all_features
from .ic_calculator import compute_forward_returns, compute_all_ic_icir, save_ic_scores
from .validator import validate_features
from .ranker import rank_signals, get_top_signals, save_ranked_signals

try:
    from core.state import AlphaLensState
except ImportError:
    # Fallback typing if core.state is not importable in this context
    AlphaLensState = dict

OHLCV_PATH = os.getenv("OHLCV_PATH", "data/processed/ohlcv.parquet")
FUNDAMENTALS_PATH = os.getenv("FUNDAMENTALS_PATH", "data/processed/fundamentals.parquet")
FEATURES_OUTPUT_PATH = os.getenv("FEATURES_OUTPUT_PATH", "data/processed/features.parquet")
VALIDATED_OUTPUT_PATH = os.getenv("VALIDATED_OUTPUT_PATH", "outputs/validated_features.parquet")


def signal_agent_node(state: "AlphaLensState") -> "AlphaLensState":
    """
    LangGraph node: full signal generation feature factory pipeline.

    Reads:
        state["signal_hypotheses"] — optional list of hypotheses from the
        Literature Agent (currently used for logging/context; the feature
        factory itself runs unconditionally across all 312 features).

    Writes:
        raw_features, validated_features, ic_scores, icir_scores, ranked_signals.
    """
    logs = list(state.get("logs", []))
    errors = list(state.get("errors", []))

    # ── Step 1: Load data ────────────────────────────────────────────────────
    try:
        ohlcv = load_ohlcv(OHLCV_PATH)
        logs.append(f"[signal_agent] Loaded OHLCV: {ohlcv.shape}")
    except Exception as e:
        errors.append(f"[signal_agent] Failed to load OHLCV: {e}")
        return {**state, "errors": errors, "logs": logs}

    fundamentals = None
    try:
        fundamentals = load_fundamentals(FUNDAMENTALS_PATH)
        logs.append(f"[signal_agent] Loaded fundamentals: {fundamentals.shape}")
    except Exception as e:
        logs.append(f"[signal_agent] No fundamentals loaded ({e}); "
                     f"value/quality/alternative features will be skipped.")

    # ── Step 2: Compute all features ─────────────────────────────────────────
    raw_features = compute_all_features(ohlcv, fundamentals)
    logs.append(f"[signal_agent] Computed {raw_features.shape[1]} raw features "
                f"across {raw_features.shape[0]} (date, ticker) rows")

    os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)
    raw_features.to_parquet(FEATURES_OUTPUT_PATH)

    # ── Step 3: Forward returns ──────────────────────────────────────────────
    fwd_returns = compute_forward_returns(ohlcv)

    # ── Step 4: IC / ICIR ─────────────────────────────────────────────────────
    ic_dict, icir_dict = compute_all_ic_icir(raw_features, fwd_returns)
    save_ic_scores(ic_dict, icir_dict)
    logs.append(f"[signal_agent] Computed IC/ICIR for {len(ic_dict)} features")

    # ── Step 5: Validation ───────────────────────────────────────────────────
    validated_features = validate_features(raw_features, ic_dict, icir_dict)
    os.makedirs(os.path.dirname(VALIDATED_OUTPUT_PATH), exist_ok=True)
    validated_features.to_parquet(VALIDATED_OUTPUT_PATH)
    logs.append(f"[signal_agent] Validated features: {validated_features.shape[1]} "
                f"of {raw_features.shape[1]} passed")

    # ── Step 6: Ranking ───────────────────────────────────────────────────────
    ranked_signals = rank_signals(icir_dict)
    save_ranked_signals(ranked_signals)
    top_signals = get_top_signals(ranked_signals, top_n=50)
    logs.append(f"[signal_agent] Top signal: {ranked_signals[0] if ranked_signals else 'N/A'} "
                f"(ICIR={icir_dict.get(ranked_signals[0], 0.0):.4f})" if ranked_signals else
                "[signal_agent] No signals ranked")

    return {
        **state,
        "raw_features": raw_features,
        "validated_features": validated_features,
        "ic_scores": ic_dict,
        "icir_scores": icir_dict,
        "ranked_signals": ranked_signals,
        "errors": errors,
        "logs": logs,
    }
