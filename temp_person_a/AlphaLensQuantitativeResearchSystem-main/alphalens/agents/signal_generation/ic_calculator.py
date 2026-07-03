"""
ic_calculator.py
Information Coefficient (IC) and IC Information Ratio (ICIR) computation —
AlphaLens Signal Generation Agent.

IC is computed cross-sectionally (per date, across the ticker universe) using
Spearman rank correlation between a feature and forward returns. ICIR is the
mean IC divided by the standard deviation of IC across time — a measure of
signal consistency, analogous to a Sharpe ratio for a forecasting signal.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple

FORWARD_RETURNS_HORIZON = 21  # 1-month forward returns


def compute_forward_returns(
    prices: pd.DataFrame,
    horizon: int = FORWARD_RETURNS_HORIZON,
) -> pd.Series:
    """
    Compute forward returns for IC calculation.

    Args:
        prices: MultiIndex (date, ticker) DataFrame with 'adj_close'.
        horizon: number of trading days forward.

    Returns:
        Series indexed by (date, ticker) named 'fwd_return'.
    """
    close = prices["adj_close"].unstack("ticker")
    fwd = close.shift(-horizon) / close - 1
    return fwd.stack().rename("fwd_return")


def compute_ic_series(
    feature: pd.Series,
    fwd_returns: pd.Series,
    min_obs: int = 10,
) -> pd.Series:
    """
    Compute cross-sectional Spearman IC per date.

    Args:
        feature: Series indexed by (date, ticker).
        fwd_returns: Series indexed by (date, ticker), same index space.
        min_obs: minimum number of tickers required on a date to compute IC.

    Returns:
        Time series of IC values, indexed by date.
    """
    combined = pd.concat([feature, fwd_returns], axis=1).dropna()
    combined.columns = ["feature", "fwd_return"]

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < min_obs:
            return np.nan
        if group["feature"].nunique() < 2 or group["fwd_return"].nunique() < 2:
            return np.nan
        r, _ = spearmanr(group["feature"], group["fwd_return"])
        return r

    return combined.groupby(level="date").apply(_spearman)


def compute_ic(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """Mean IC across dates."""
    ic_series = compute_ic_series(feature, fwd_returns)
    if ic_series.dropna().empty:
        return 0.0
    return float(ic_series.mean())


def compute_icir(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC)."""
    ic_series = compute_ic_series(feature, fwd_returns).dropna()
    if ic_series.empty or ic_series.std() == 0 or np.isnan(ic_series.std()):
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def compute_all_ic_icir(
    features: pd.DataFrame,
    fwd_returns: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute IC and ICIR for all features.

    Args:
        features: MultiIndex (date, ticker) DataFrame, one column per feature.
        fwd_returns: Series indexed by (date, ticker).

    Returns:
        (ic_dict, icir_dict) — feature name -> float.
    """
    ic_dict: Dict[str, float] = {}
    icir_dict: Dict[str, float] = {}

    for col in features.columns:
        try:
            ic_dict[col] = compute_ic(features[col], fwd_returns)
            icir_dict[col] = compute_icir(features[col], fwd_returns)
        except Exception as e:
            print(f"[IC_CALCULATOR] Failed for feature '{col}': {e}")
            ic_dict[col] = 0.0
            icir_dict[col] = 0.0

    return ic_dict, icir_dict


def save_ic_scores(
    ic_dict: Dict[str, float],
    icir_dict: Dict[str, float],
    ic_path: str = "outputs/ic_scores.json",
    icir_path: str = "outputs/icir_scores.json",
) -> None:
    """Persist IC and ICIR dicts as JSON."""
    import json
    import os

    os.makedirs(os.path.dirname(ic_path), exist_ok=True)
    with open(ic_path, "w") as f:
        json.dump(ic_dict, f, indent=2)
    with open(icir_path, "w") as f:
        json.dump(icir_dict, f, indent=2)
