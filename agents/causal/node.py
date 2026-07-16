import json

import pandas as pd

from agents.causal.data_prep import build_causal_dataset, select_uncorrelated_top_signals, subsample_non_overlapping
from agents.causal.dag_builder import build_domain_dag
from agents.causal.dml import estimate_all_ates, select_causal_signals

from core.state import AlphaLensState


def causal_agent_node(state: AlphaLensState) -> AlphaLensState:
    """LangGraph node: full causal inference pipeline."""
    logs = list(state.get("logs", []))
    errors = list(state.get("errors", []))

    logs.append("causal_agent: loading features and computing forward returns")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    features = features.set_index(["date", "ticker"]).sort_index()

    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "ticker"]).sort_index()
    close = prices["adj_close"].unstack("ticker")
    fwd_returns = (close.shift(-21) / close - 1).stack().rename("fwd_return")

    with open("outputs/ranked_signals.json") as f:
        ranked = json.load(f)

    top_signals = select_uncorrelated_top_signals(ranked, features, top_n=10, max_corr=0.95)
    logs.append(f"causal_agent: selected {len(top_signals)} de-duplicated signals")

    causal_df = build_causal_dataset(features, fwd_returns, top_signals)
    causal_df = subsample_non_overlapping(causal_df, horizon=21)
    logs.append(f"causal_agent: built causal dataset (de-overlapped), shape={causal_df.shape}")

    dag = build_domain_dag(top_signals)
    dag_structure = {"nodes": list(dag.nodes()), "edges": list(dag.edges())}
    logs.append(f"causal_agent: built domain DAG, {len(dag_structure['edges'])} edges")

    ate_results = estimate_all_ates(causal_df, top_signals)
    causal_signals = select_causal_signals(ate_results)
    logs.append(f"causal_agent: estimated ATEs for {len(ate_results)} signals, {len(causal_signals)} causally significant")

    return {
        **state,
        "dag_structure": dag_structure,
        "ate_estimates": ate_results,
        "causal_signals": causal_signals,
        "logs": logs,
        "errors": errors,
    }


if __name__ == "__main__":
    test_state: AlphaLensState = {"run_id": "test-003", "universe": [], "as_of_date": "2026-07-15", "errors": [], "logs": []}

    print("Running full causal_agent_node end-to-end...\n")
    result = causal_agent_node(test_state)

    print("\n=== RESULT SUMMARY ===")
    print(f"DAG edges: {len(result['dag_structure']['edges'])}")
    print(f"ATE estimates: {len(result['ate_estimates'])}")
    print(f"Causal signals: {result['causal_signals']}")

    for line in result["logs"]:
        print(f"  - {line}")

    assert "ate_estimates" in result and len(result["ate_estimates"]) > 0
    print("\nPASS: causal_agent_node completed successfully")