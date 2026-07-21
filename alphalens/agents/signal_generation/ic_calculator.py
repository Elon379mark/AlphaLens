"""
ic_calculator.py
Information Coefficient (IC) and IC Information Ratio (ICIR) computation —
AlphaLens Signal Generation Agent.

IC is computed cross-sectionally (per date, across the ticker universe) using
Spearman rank correlation between a feature and forward returns. ICIR is the
mean IC divided by the standard deviation of IC across time — a measure of
signal consistency, analogous to a Sharpe ratio for a forecasting signal.
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

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


def compute_ic_series(feature: pd.Series, fwd_returns: pd.Series, min_obs: int = 3) -> pd.Series:
    """
    Computes Information Coefficient (Spearman rank correlation) time series.
    Vectorized implementation for fast performance across thousands of dates.
    """
    combined = pd.concat([feature, fwd_returns], axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    combined.columns = ["feature", "fwd_return"]

    # Compute ranks per date
    ranks = combined.groupby(level="date").rank()
    
    # Vectorized Pearson correlation of ranks per date == Spearman rank correlation
    def _fast_corr(group):
        if len(group) < min_obs:
            return np.nan
        cov = np.cov(group["feature"], group["fwd_return"])
        var_f = cov[0, 0]
        var_r = cov[1, 1]
        if var_f < 1e-12 or var_r < 1e-12:
            return np.nan
        return cov[0, 1] / np.sqrt(var_f * var_r)

    return ranks.groupby(level="date").apply(_fast_corr, include_groups=False)


def compute_ic(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """Mean IC across dates."""
    ic_series = compute_ic_series(feature, fwd_returns)
    if ic_series.dropna().empty:
        return 0.0
    return float(ic_series.mean())


def compute_icir(feature: pd.Series, fwd_returns: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC)."""
    ic_series = compute_ic_series(feature, fwd_returns)
    valid_ic = ic_series.dropna()
    if valid_ic.empty or valid_ic.std() == 0:
        return 0.0
    return float(valid_ic.mean() / valid_ic.std())


def compute_all_ic_icir(features: pd.DataFrame, fwd_returns: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fast parallelized IC/ICIR calculation across all feature columns.
    """
    ic_dict = {}
    icir_dict = {}

    for col in features.columns:
        s = compute_ic_series(features[col], fwd_returns)
        v = s.dropna()
        if v.empty:
            ic_dict[col] = 0.0
            icir_dict[col] = 0.0
        else:
            mean_ic = float(v.mean())
            std_ic = float(v.std())
            ic_dict[col] = mean_ic
            icir_dict[col] = mean_ic / std_ic if std_ic > 1e-12 else 0.0

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
