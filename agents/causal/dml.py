from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor


def run_linear_dml(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str = "fwd_return",
    confounder_cols: Optional[List[str]] = None,
) -> Dict:
    """
    DML with linear final stage. Returns {ate, ci_low, ci_high}.
    """
    confounder_cols = confounder_cols or []
    W = df[confounder_cols].values if confounder_cols else None
    T = df[treatment_col].values
    Y = df[outcome_col].values

    model = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        discrete_treatment=True,  # our treatments are binary (0/1), not continuous — important distinction from the manual's default
        cv=3,  # reduced from manual's 5 given our small dataset size
        random_state=42,
    )
    model.fit(Y, T, X=None, W=W)

    ate = float(model.ate())
    ci = model.ate_interval(alpha=0.05)
    return {"ate": ate, "ci_low": float(ci[0]), "ci_high": float(ci[1])}


def estimate_all_ates(df: pd.DataFrame, signals: List[str], confounders: Optional[List[str]] = None) -> Dict[str, dict]:
    """Estimate ATE for each signal's treatment indicator."""
    results = {}
    for i, sig in enumerate(signals, 1):
        treatment_col = f"{sig}_treated"
        if treatment_col not in df.columns:
            continue
        print(f"  [{i}/{len(signals)}] Estimating ATE for {sig}...")
        try:
            result = run_linear_dml(df, treatment_col=treatment_col, confounder_cols=confounders)
            results[sig] = result
            print(f"    ATE={result['ate']:.5f}  CI=[{result['ci_low']:.5f}, {result['ci_high']:.5f}]")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[sig] = {"ate": None, "ci_low": None, "ci_high": None, "error": str(e)}
    return results


def select_causal_signals(ate_results: Dict[str, dict], significance_level: float = 0.05) -> List[str]:
    """Select signals with statistically significant positive ATE (CI excludes zero, ATE > 0)."""
    causal = []
    for sig, r in ate_results.items():
        if r.get("ate") is None:
            continue
        if r["ci_low"] > 0 or r["ci_high"] < 0:
            if r["ate"] > 0:
                causal.append(sig)
    return causal


if __name__ == "__main__":
    import json
    from agents.causal.data_prep import build_causal_dataset

    print("Loading data...")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    features = features.set_index(["date", "ticker"]).sort_index()

    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "ticker"]).sort_index()
    close = prices["adj_close"].unstack("ticker")
    fwd_returns = (close.shift(-21) / close - 1).stack().rename("fwd_return")

    from agents.causal.data_prep import select_uncorrelated_top_signals, subsample_non_overlapping

    with open("outputs/ranked_signals.json") as f:
       ranked = json.load(f)
    print("Selecting top 10 signals, skipping near-duplicates (|corr| > 0.95)...")
    top_signals = select_uncorrelated_top_signals(ranked, features, top_n=10, max_corr=0.95)
    print(f"Final selected signals: {top_signals}")

    causal_df = build_causal_dataset(features, fwd_returns, top_signals)
    print(f"Causal dataset shape before de-overlapping: {causal_df.shape}")

    causal_df = subsample_non_overlapping(causal_df, horizon=21)
    print(f"Causal dataset shape AFTER removing overlapping windows: {causal_df.shape}")
    print("(This is expected to be ~1/21st the size — each ticker now contributes")
    print(" one observation per ~21-day window instead of one per day, removing")
    print(" the autocorrelation that was producing artificially tight, all-significant CIs)")

    print(f"\nEstimating ATE for {len(top_signals)} signals via Double ML (this will take several minutes)...\n")
    ate_results = estimate_all_ates(causal_df, top_signals)

    print(f"\n{'='*50}")
    print("ATE RESULTS SUMMARY")
    print(f"{'='*50}")
    for sig, r in ate_results.items():
        if r.get("ate") is not None:
            print(f"{sig}: ATE={r['ate']:.5f}, CI=[{r['ci_low']:.5f}, {r['ci_high']:.5f}]")
        else:
            print(f"{sig}: FAILED - {r.get('error', 'unknown')}")

    causal_signals = select_causal_signals(ate_results)
    print(f"\nSignificant causal signals (CI excludes zero, ATE>0): {causal_signals}")
    print("(On synthetic random-walk data, expect FEW OR ZERO significant results — this is correct, not a bug)")

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ate_estimates.json", "w") as f:
        json.dump(ate_results, f, indent=2)
    with open("outputs/causal_signals.json", "w") as f:
        json.dump(causal_signals, f, indent=2)

    n_succeeded = sum(1 for r in ate_results.values() if r.get("ate") is not None)
    assert n_succeeded > 0, "All DML estimations failed — check errors above"
    print(f"\nPASS: DML completed for {n_succeeded}/{len(top_signals)} signals")