"""
validator.py
Feature Validation Pipeline — AlphaLens Signal Generation Agent

Filters the raw 312-feature set down to a validated subset based on:
  1. NaN fraction threshold (data sufficiency)
  2. Minimum absolute IC (predictive power)
  3. Minimum absolute ICIR (consistency over time)

Also provides correlation-based redundancy detection for further pruning.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

IC_THRESHOLD = 0.02      # Minimum absolute IC
ICIR_THRESHOLD = 0.5     # Minimum absolute ICIR
NAN_THRESHOLD = 0.30     # Max fraction of NaN values allowed


def validate_features(
    features: pd.DataFrame,
    ic_dict: Dict[str, float],
    icir_dict: Dict[str, float],
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
    nan_threshold: float = NAN_THRESHOLD,
) -> pd.DataFrame:
    """
    Remove features that fail validation criteria.

    A feature is kept only if:
      1. |IC| >= ic_threshold
      2. |ICIR| >= icir_threshold
      3. NaN fraction <= nan_threshold

    Args:
        features: MultiIndex (date, ticker) DataFrame, one column per feature.
        ic_dict: feature name -> mean IC.
        icir_dict: feature name -> ICIR.
        ic_threshold, icir_threshold, nan_threshold: validation cutoffs.

    Returns:
        Filtered DataFrame containing only validated feature columns.
    """
    valid_cols = []
    for col in features.columns:
        nan_frac = features[col].isna().mean()
        ic = abs(ic_dict.get(col, 0.0))
        icir = abs(icir_dict.get(col, 0.0))
        if nan_frac <= nan_threshold and ic >= ic_threshold and icir >= icir_threshold:
            valid_cols.append(col)

    logger.info(f"{len(valid_cols)}/{len(features.columns)} features passed.")
    return features[valid_cols]


def check_feature_correlation(
    features: pd.DataFrame,
    max_corr: float = 0.95,
) -> List[str]:
    """
    Identify features with high pairwise correlation (redundancy candidates).

    Args:
        features: DataFrame of features (any index).
        max_corr: correlation threshold above which features are flagged.

    Returns:
        List of feature names whose max pairwise correlation exceeds max_corr.
        Caller should keep one from each correlated pair/cluster and drop the rest.
    """
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [col for col in upper.columns if any(upper[col] > max_corr)]
    return redundant


def get_validation_summary(
    features: pd.DataFrame,
    ic_dict: Dict[str, float],
    icir_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Produce a per-feature validation summary table for inspection/debugging.

    Returns:
        DataFrame indexed by feature name with columns:
        nan_fraction, ic, icir, passed.
    """
    rows = []
    for col in features.columns:
        nan_frac = features[col].isna().mean()
        ic = ic_dict.get(col, 0.0)
        icir = icir_dict.get(col, 0.0)
        passed = (
            nan_frac <= NAN_THRESHOLD
            and abs(ic) >= IC_THRESHOLD
            and abs(icir) >= ICIR_THRESHOLD
        )
        rows.append({
            "feature": col,
            "nan_fraction": nan_frac,
            "ic": ic,
            "icir": icir,
            "passed": passed,
        })
    summary = pd.DataFrame(rows).set_index("feature")
    return summary.sort_values("icir", key=lambda s: s.abs(), ascending=False)
