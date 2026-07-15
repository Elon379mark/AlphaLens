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

def select_uncorrelated_top_signals(
    ranked_signals: List[str],
    features: pd.DataFrame,
    top_n: int = 10,
    max_corr: float = 0.95,
) -> List[str]:
    """
    Walk down the ranked signal list, greedily selecting signals that are
    NOT highly correlated (|corr| > max_corr) with any signal already
    selected. This prevents near-duplicate columns (e.g. a raw ratio and
    its z-score transform) from both being selected, which causes a
    singular correlation matrix in PC algorithm's Fisher-Z test and
    silently distorts DML results by duplicating the same information
    as two "different" treatments.
    """
    selected: List[str] = []
    for sig in ranked_signals:
        if len(selected) >= top_n:
            break
        if sig not in features.columns:
            continue
        candidate_series = features[sig]
        is_redundant = False
        for chosen in selected:
            corr = candidate_series.corr(features[chosen])
            if pd.notna(corr) and abs(corr) > max_corr:
                is_redundant = True
                print(f"  Skipping '{sig}' — corr={corr:.3f} with already-selected '{chosen}'")
                break
        if not is_redundant:
            selected.append(sig)
    return selected
def subsample_non_overlapping(causal_df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    causal_df is indexed by (date, ticker). Because fwd_return is a 21-day
    forward return computed on EVERY day, consecutive rows for the same
    ticker share ~20/21 of their underlying window — heavy autocorrelation
    that violates the (approximate) independence DML's confidence intervals
    assume, producing artificially tight CIs and spurious "significant"
    results. Keeping only every `horizon`-th observation per ticker removes
    most of this overlap, giving a more honest (if smaller) dataset for
    causal estimation. This mirrors the "purged/embargo" logic referenced
    in the project manual's own backtesting section.
    """
    df = causal_df.copy()
    df["_day_rank"] = df.groupby(level="ticker").cumcount()
    df = df[df["_day_rank"] % horizon == 0]
    df = df.drop(columns=["_day_rank"])
    return df
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
    print("Selecting top 10 signals, skipping near-duplicates (|corr| > 0.95)...")
    top_signals = select_uncorrelated_top_signals(ranked, features, top_n=10, max_corr=0.95)
    print(f"Final selected signals: {top_signals}")

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