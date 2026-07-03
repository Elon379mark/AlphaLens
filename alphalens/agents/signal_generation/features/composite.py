"""
composite.py
Composite Features — AlphaLens Signal Generation Agent
~12 features combining signals across categories: factor blends, PCA
factors, and risk-adjusted composites.

These features require the full 300-feature set as input (momentum, value,
quality, volatility, volume, technical, alternative already computed) and
are calculated last in the pipeline.
"""

import pandas as pd
import numpy as np


def compute_composite_features(all_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite (combo) features from the already-built feature matrix.

    Args:
        all_features: MultiIndex (date, ticker) DataFrame containing momentum,
                       value, quality, volatility, volume, technical, and
                       alternative features.

    Returns:
        DataFrame with ~12 composite feature columns.
    """
    feat: dict = {}

    def _safe_col(name: str) -> pd.Series | None:
        return all_features[name] if name in all_features.columns else None

    def _cs_rank(series: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank per date."""
        return series.groupby(level="date").rank(pct=True)

    # ── 1. Momentum-Quality combo ─────────────────────────────────────────────
    mom = _safe_col("mom_12_1")
    qual = _safe_col("quality_composite")
    if mom is not None and qual is not None:
        feat["mom_quality_combo"] = (_cs_rank(mom) + _cs_rank(qual)) / 2

    # ── 2. Value-Quality combo (classic "quality at a reasonable price") ─────
    val = _safe_col("value_composite")
    if val is not None and qual is not None:
        feat["value_quality_combo"] = (_cs_rank(val) + _cs_rank(qual)) / 2

    # ── 3. Momentum-Value combo ───────────────────────────────────────────────
    if mom is not None and val is not None:
        feat["mom_value_combo"] = (_cs_rank(mom) + _cs_rank(val)) / 2

    # ── 4. Low-volatility quality combo (defensive factor) ──────────────────
    vol60 = _safe_col("vol_60d")
    if vol60 is not None and qual is not None:
        feat["low_vol_quality_combo"] = (_cs_rank(-vol60) + _cs_rank(qual)) / 2

    # ── 5. Risk-adjusted momentum (momentum / volatility) ────────────────────
    if mom is not None and vol60 is not None:
        feat["risk_adj_momentum"] = mom / (vol60 + 1e-8)

    # ── 6. Risk-adjusted value ─────────────────────────────────────────────────
    if val is not None and vol60 is not None:
        feat["risk_adj_value"] = val / (vol60 + 1e-8)

    # ── 7. Three-factor composite: momentum + value + quality ───────────────
    if mom is not None and val is not None and qual is not None:
        feat["three_factor_composite"] = (
            _cs_rank(mom) + _cs_rank(val) + _cs_rank(qual)
        ) / 3

    # ── 8. Liquidity-adjusted momentum ───────────────────────────────────────
    amihud = _safe_col("amihud_60d")
    if mom is not None and amihud is not None:
        feat["liquidity_adj_momentum"] = _cs_rank(mom) - 0.3 * _cs_rank(amihud)

    # ── 9. PCA-style composite of all rank-based features (orthogonalized) ──
    rank_like_cols = [c for c in all_features.columns if c.startswith(("rank_", "zscore_"))]
    if len(rank_like_cols) >= 5:
        sub = all_features[rank_like_cols].copy()
        # Cross-sectional standardize each date, then average (simple PC1 proxy)
        standardized = sub.groupby(level="date").transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        feat["pca_composite_proxy"] = standardized.mean(axis=1)

    # ── 10. Technical-momentum confirmation combo ────────────────────────────
    rsi14 = _safe_col("rsi_14")
    if mom is not None and rsi14 is not None:
        feat["tech_mom_confirm"] = (
            (mom > 0).astype(float) * (rsi14 > 50).astype(float)
            - (mom < 0).astype(float) * (rsi14 < 50).astype(float)
        )

    # ── 11. Alpha signal confidence score (vote count across categories) ────
    direction_cols = []
    for c in ["mom_12_1", "value_composite", "quality_composite"]:
        s = _safe_col(c)
        if s is not None:
            direction_cols.append(np.sign(s))
    if len(direction_cols) >= 2:
        votes = pd.concat(direction_cols, axis=1).sum(axis=1)
        feat["multi_factor_vote"] = votes

    # ── 12. Crowding-adjusted composite (penalize crowded shorts/longs) ─────
    short_proxy = _safe_col("short_interest_proxy")
    if mom is not None and short_proxy is not None:
        feat["crowding_adj_momentum"] = _cs_rank(mom) - 0.2 * _cs_rank(short_proxy)

    if not feat:
        return pd.DataFrame(index=all_features.index)

    result = pd.concat(feat, axis=1)
    return result
