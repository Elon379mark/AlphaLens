import json
from typing import List, Optional

import pandas as pd


def build_causal_dataset(
    features: pd.DataFrame,
    fwd_returns: pd.Series,
    top_signals: List[str],
    confounders: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build panel dataset for causal analysis.
    Includes: treatment signals, outcome (fwd_return), confounders,
    and binarized treatment indicators (top vs. bottom quintile per signal).
    """
    df = features[top_signals].copy()
    df["fwd_return"] = fwd_returns

    if confounders:
        for c in confounders:
            if c in features.columns:
                df[c] = features[c]

    for sig in top_signals:
        # Binarize: top quintile (80th percentile+) = treated, rest = control.
        # This means bottom 80% (not just bottom 20%) counts as "control" —
        # a deliberate simplification matching the manual's threshold approach,
        # not a true top-vs-bottom-quintile comparison. Worth knowing if
        # results look different from what "quintile vs quintile" would imply.
        threshold = df[sig].quantile(0.8)
        df[f"{sig}_treated"] = (df[sig] > threshold).astype(int)

    return df.dropna()


if __name__ == "__main__":
    print("Loading features and computing forward returns...")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    features = features.set_index(["date", "ticker"]).sort_index()

    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "ticker"]).sort_index()
    close = prices["adj_close"].unstack("ticker")
    fwd_returns = (close.shift(-21) / close - 1).stack().rename("fwd_return")

    print("Loading top signals from Chapter 3...")
    with open("outputs/ranked_signals.json") as f:
        ranked = json.load(f)
    top_signals = ranked[:10]  # keep small for causal chapter — DML per signal is expensive
    print(f"Using top 10 signals: {top_signals}")

    print("\nBuilding causal dataset...")
    causal_df = build_causal_dataset(features, fwd_returns, top_signals)

    print(f"\nCausal dataset shape: {causal_df.shape}")
    print(f"Columns: {list(causal_df.columns)}")

    for sig in top_signals:
        treated_frac = causal_df[f"{sig}_treated"].mean()
        print(f"  {sig}_treated: {treated_frac:.1%} treated")

    assert causal_df.shape[0] > 0, "Causal dataset is empty"
    for sig in top_signals:
        n_treated = causal_df[f"{sig}_treated"].sum()
        assert n_treated > 0, f"No treated units for {sig} — check quantile threshold"
        assert n_treated < len(causal_df), f"All units treated for {sig} — no variation, DML will fail"

    print("\nPASS: causal dataset built with valid treatment variation for all signals")