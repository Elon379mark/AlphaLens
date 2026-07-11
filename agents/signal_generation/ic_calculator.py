import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FORWARD_RETURNS_HORIZON = 21  # ~1 month forward returns


def compute_forward_returns(prices: pd.DataFrame, horizon: int = FORWARD_RETURNS_HORIZON) -> pd.Series:
    """
    Compute forward returns for IC calculation.
    Input: prices DataFrame indexed by (date, ticker), needs adj_close.
    Returns: Series indexed by (date, ticker), forward return over `horizon` days.
    """
    close = prices["adj_close"].unstack("ticker")
    fwd = close.shift(-horizon) / close - 1
    return fwd.stack().rename("fwd_return")


def compute_ic_series(feature: pd.Series, fwd_returns: pd.Series) -> pd.Series:
    """
    Compute cross-sectional Spearman IC per date.
    Returns a time series of IC values, one per date.
    """
    combined = pd.concat([feature, fwd_returns], axis=1).dropna()
    combined.columns = ["feature", "fwd_return"]

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 10:
            return np.nan
        r, _ = spearmanr(group["feature"], group["fwd_return"])
        return r

    return combined.groupby(level="date").apply(_spearman)


def compute_ic(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """Mean IC across all dates."""
    ic_series = compute_ic_series(feature, fwd_returns)
    return float(ic_series.mean()) if not ic_series.empty else 0.0


def compute_icir(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC)."""
    ic_series = compute_ic_series(feature, fwd_returns).dropna()
    if len(ic_series) < 2 or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def compute_all_ic_icir(features: pd.DataFrame, fwd_returns: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute IC and ICIR for every feature column.
    Returns (ic_dict, icir_dict), keyed by feature name.
    """
    ic_dict = {}
    icir_dict = {}
    total = len(features.columns)

    for i, col in enumerate(features.columns, 1):
        ic_dict[col] = compute_ic(features[col], fwd_returns)
        icir_dict[col] = compute_icir(features[col], fwd_returns)
        if i % 50 == 0 or i == total:
            print(f"  Processed {i}/{total} features...")

    return ic_dict, icir_dict


def save_ic_icir(ic_dict: Dict[str, float], icir_dict: Dict[str, float],
                  ic_path: str = "outputs/ic_scores.json",
                  icir_path: str = "outputs/icir_scores.json") -> None:
    Path(ic_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ic_path, "w") as f:
        json.dump(ic_dict, f, indent=2)
    with open(icir_path, "w") as f:
        json.dump(icir_dict, f, indent=2)
    print(f"Saved IC scores to {ic_path}")
    print(f"Saved ICIR scores to {icir_path}")


if __name__ == "__main__":
    from agents.signal_generation.data_loader import load_ohlcv

    print("Loading features...")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    features = features.set_index(["date", "ticker"]).sort_index()
    print(f"Features shape: {features.shape}")

    print("\nLoading prices for forward returns...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")

    print(f"\nComputing forward returns ({FORWARD_RETURNS_HORIZON}-day horizon)...")
    fwd_returns = compute_forward_returns(prices)
    print(f"Forward returns shape: {fwd_returns.shape}")
    print(f"Non-null forward returns: {fwd_returns.notna().sum()} / {len(fwd_returns)}")

    print("\nComputing IC/ICIR for all 312 features (this will take a couple minutes)...")
    ic_dict, icir_dict = compute_all_ic_icir(features, fwd_returns)

    save_ic_icir(ic_dict, icir_dict)

    # --- Summary stats ---
    ic_series = pd.Series(ic_dict).dropna()
    icir_series = pd.Series(icir_dict).dropna()

    print(f"\n{'='*50}")
    print("IC SUMMARY")
    print(f"{'='*50}")
    print(f"Mean |IC| across all features: {ic_series.abs().mean():.4f}")
    print(f"Max |IC|: {ic_series.abs().max():.4f} ({ic_series.abs().idxmax()})")
    print(f"Features with |IC| >= 0.02: {(ic_series.abs() >= 0.02).sum()} / {len(ic_series)}")

    print(f"\n{'='*50}")
    print("ICIR SUMMARY")
    print(f"{'='*50}")
    print(f"Mean |ICIR| across all features: {icir_series.abs().mean():.4f}")
    print(f"Max |ICIR|: {icir_series.abs().max():.4f} ({icir_series.abs().idxmax()})")
    print(f"Features with |ICIR| >= 0.5: {(icir_series.abs() >= 0.5).sum()} / {len(icir_series)}")

    print(f"\nTop 10 features by |IC|:")
    top10 = ic_series.abs().sort_values(ascending=False).head(10)
    for name, val in top10.items():
        print(f"  {name}: {ic_dict[name]:.4f}")

    assert len(ic_dict) == 312, f"Expected IC for 312 features, got {len(ic_dict)}"
    assert len(icir_dict) == 312, f"Expected ICIR for 312 features, got {len(icir_dict)}"
    print("\nPASS: IC and ICIR computed for all 312 features")