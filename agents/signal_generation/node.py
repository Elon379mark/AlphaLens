from typing import Dict

from agents.signal_generation.build_features import build_all_features
from agents.signal_generation.ic_calculator import compute_forward_returns, compute_all_ic_icir, save_ic_icir
from agents.signal_generation.validator import validate_features, save_validated_features
from agents.signal_generation.ranker import rank_signals, get_top_signals
from agents.signal_generation.data_loader import load_ohlcv

from core.state import AlphaLensState

OHLCV_PATH = "data/processed/sample_prices.parquet"
FUNDAMENTALS_PATH = "data/processed/sample_fundamentals.parquet"


def signal_agent_node(state: AlphaLensState) -> AlphaLensState:
    """
    LangGraph node: full signal generation pipeline.
    Builds all 312 features, computes IC/ICIR, validates, and ranks signals.
    """
    logs = list(state.get("logs", []))
    errors = list(state.get("errors", []))

    # --- Build all 312 features ---
    logs.append("signal_agent: building feature matrix")
    try:
        raw_features = build_all_features(OHLCV_PATH, FUNDAMENTALS_PATH)
        logs.append(f"signal_agent: built {raw_features.shape[1]} features across {raw_features.shape[0]} rows")
    except Exception as e:
        errors.append(f"signal_agent: feature build failed: {e}")
        return {**state, "logs": logs, "errors": errors}

    # --- Forward returns + IC/ICIR ---
    logs.append("signal_agent: computing forward returns and IC/ICIR")
    prices = load_ohlcv(OHLCV_PATH)
    fwd_returns = compute_forward_returns(prices)
    ic_dict, icir_dict = compute_all_ic_icir(raw_features, fwd_returns)
    save_ic_icir(ic_dict, icir_dict)
    logs.append(f"signal_agent: computed IC/ICIR for {len(ic_dict)} features")

    # --- Validation ---
    logs.append("signal_agent: validating features")
    validated_features, rejections = validate_features(raw_features, ic_dict, icir_dict)
    save_validated_features(validated_features)
    logs.append(f"signal_agent: {validated_features.shape[1]}/{raw_features.shape[1]} features passed validation")

    # --- Ranking (rank ALL signals, not just validated ones, so downstream
    #     agents can see relative strength even among rejected features) ---
    logs.append("signal_agent: ranking signals")
    ranked_signals = rank_signals(icir_dict)
    logs.append(f"signal_agent: ranked {len(ranked_signals)} signals")

    return {
        **state,
        "raw_features": raw_features,
        "validated_features": validated_features,
        "ic_scores": ic_dict,
        "icir_scores": icir_dict,
        "ranked_signals": ranked_signals,
        "logs": logs,
        "errors": errors,
    }


if __name__ == "__main__":
    test_state: AlphaLensState = {
        "run_id": "test-run-002",
        "universe": [],
        "as_of_date": "2026-07-11",
        "errors": [],
        "logs": [],
    }

    print("Running full signal_agent_node end-to-end (this will take several minutes — ")
    print("it runs all 8 feature categories, IC/ICIR for 312 features, validation, and ranking)...\n")

    result = signal_agent_node(test_state)

    print("\n=== RESULT SUMMARY ===")
    print(f"Raw features: {result['raw_features'].shape if 'raw_features' in result else 'MISSING'}")
    print(f"Validated features: {result['validated_features'].shape if 'validated_features' in result else 'MISSING'}")
    print(f"IC scores computed: {len(result.get('ic_scores', {}))}")
    print(f"ICIR scores computed: {len(result.get('icir_scores', {}))}")
    print(f"Ranked signals: {len(result.get('ranked_signals', []))}")

    print(f"\nTop 5 ranked signals:")
    for name in result.get("ranked_signals", [])[:5]:
        print(f"  {name}: IC={result['ic_scores'][name]:.4f}, ICIR={result['icir_scores'][name]:.4f}")

    print(f"\nLogs:")
    for line in result["logs"]:
        print(f"  - {line}")
    if result["errors"]:
        print(f"\nErrors:")
        for line in result["errors"]:
            print(f"  - {line}")

    assert "raw_features" in result and result["raw_features"].shape[1] == 312, "raw_features missing or wrong shape"
    assert "ranked_signals" in result and len(result["ranked_signals"]) == 312, "ranked_signals missing or wrong length"
    print("\nPASS: signal_agent_node completed successfully.")