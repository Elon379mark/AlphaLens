"""
AlphaLens — Signal Generation Features
Exports a single compute_all_features() function that runs all 8 feature modules
and returns a flat DataFrame of 312 columns.
"""

import pandas as pd
from .momentum import compute_momentum_features
from .value import compute_value_features
from .quality import compute_quality_features
from .volatility import compute_volatility_features
from .volume import compute_volume_features
from .technical import compute_technical_features
from .alternative import compute_alternative_features
from .composite import compute_composite_features


def compute_all_features(
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all 312 alpha features.

    Args:
        ohlcv: MultiIndex (date, ticker) DataFrame with adj_close, volume, returns.
        fundamentals: MultiIndex (date, ticker) DataFrame with fundamental columns.
                      If None, value/quality/alternative features are skipped.

    Returns:
        DataFrame indexed by (date, ticker) with 312 feature columns.
    """
    feature_dfs = []

    # Group 1: Momentum (~60 features) — OHLCV only
    mom = compute_momentum_features(ohlcv)
    feature_dfs.append(mom)

    # Group 2: Volatility (~40 features) — OHLCV only
    vol = compute_volatility_features(ohlcv)
    feature_dfs.append(vol)

    # Group 3: Volume (~35 features) — OHLCV only
    volume = compute_volume_features(ohlcv)
    feature_dfs.append(volume)

    # Group 4: Technical (~40 features) — OHLCV only
    tech = compute_technical_features(ohlcv)
    feature_dfs.append(tech)

    if fundamentals is not None:
        # Group 5: Value (~50 features)
        val = compute_value_features(ohlcv, fundamentals)
        feature_dfs.append(val)

        # Group 6: Quality (~45 features)
        qual = compute_quality_features(fundamentals)
        feature_dfs.append(qual)

        # Group 7: Alternative (~30 features)
        alt = compute_alternative_features(ohlcv, fundamentals)
        feature_dfs.append(alt)

    # Merge all on common index
    all_features = pd.concat(feature_dfs, axis=1)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    if fundamentals is not None:
        # Group 8: Composite (~12 features) — needs all prior features
        comp = compute_composite_features(all_features)
        all_features = pd.concat([all_features, comp], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    # Drop fully-empty columns and return
    all_features = all_features.dropna(axis=1, how="all")
    return all_features


__all__ = [
    "compute_all_features",
    "compute_momentum_features",
    "compute_value_features",
    "compute_quality_features",
    "compute_volatility_features",
    "compute_volume_features",
    "compute_technical_features",
    "compute_alternative_features",
    "compute_composite_features",
]
