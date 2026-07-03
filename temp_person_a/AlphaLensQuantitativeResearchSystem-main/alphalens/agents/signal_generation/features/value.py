"""
value.py
Value Features — AlphaLens Signal Generation Agent
~50 features capturing relative cheapness of equities.

Features:
  - Classic ratios: PE, PB, EV/EBITDA, Earnings Yield, Div Yield
  - Cross-sectional ranks and z-scores of ratios
  - Ratio trends (change in ratio over time)
  - Composite value scores
  - Price-to-sales, Price-to-cashflow proxies
  - Debt-adjusted value metrics
"""

import pandas as pd
import numpy as np


def compute_value_features(
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute value features.

    Args:
        ohlcv: MultiIndex (date, ticker) with adj_close.
        fundamentals: MultiIndex (date, ticker) with fundamental columns.

    Returns:
        DataFrame with value feature columns.
    """
    # Align fundamentals to OHLCV dates via forward-fill.
    # IMPORTANT: reindexing to the OHLCV date index *before* ffill would drop
    # every actual fundamentals observation (since quarterly report dates
    # rarely coincide exactly with OHLCV trading dates), leaving nothing to
    # forward-fill from. Instead we take the union of both date sets, ffill
    # across that union, then select down to the OHLCV dates.
    ohlcv_dates = ohlcv.unstack("ticker").index
    fun_wide = fundamentals.unstack("ticker")
    union_dates = fun_wide.index.union(ohlcv_dates).sort_values()
    fun = (
        fun_wide
        .reindex(union_dates)
        .ffill()
        .reindex(ohlcv_dates)
        .stack("ticker")
        .reindex(ohlcv.index)
    )

    feat: dict = {}

    # ── 1. Raw fundamental ratios (6 features) ───────────────────────────────
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda", "earnings_yield",
                "dividend_yield", "debt_equity"]:
        if col in fun.columns:
            feat[f"raw_{col}"] = fun[col].unstack("ticker")

    # ── 2. Inverted ratios (higher = cheaper) (3 features) ───────────────────
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
        if col in fun.columns:
            series = fun[col].unstack("ticker")
            feat[f"inv_{col}"] = 1.0 / series.replace(0, np.nan)

    # ── 3. Cross-sectional rank of ratios (6 features) ───────────────────────
    for col in ["earnings_yield", "dividend_yield"]:
        if col in fun.columns:
            raw = fun[col].unstack("ticker")
            feat[f"rank_{col}"] = raw.rank(axis=1, pct=True)

    for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
        if col in fun.columns:
            raw = fun[col].unstack("ticker")
            # Lower ratio = better value → rank ascending (rank 1.0 = cheapest)
            feat[f"rank_{col}"] = (1 - raw.rank(axis=1, pct=True))

    # ── 4. Z-score of ratios (cross-sectional) (6 features) ──────────────────
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda", "earnings_yield",
                "dividend_yield", "debt_equity"]:
        if col in fun.columns:
            raw = fun[col].unstack("ticker")
            cs_mean = raw.mean(axis=1)
            cs_std  = raw.std(axis=1).replace(0, np.nan)
            feat[f"zscore_{col}"] = raw.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # ── 5. Ratio change (trend) (6 features) ─────────────────────────────────
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
        if col in fun.columns:
            raw = fun[col].unstack("ticker")
            feat[f"delta_{col}_63d"]  = raw.pct_change(63)
            feat[f"delta_{col}_252d"] = raw.pct_change(252)

    # ── 6. Composite value scores (4 features) ───────────────────────────────
    # Equal-weight composite of rank scores. Each rank_* entry is a wide
    # (date x ticker) DataFrame with identical shape/index, so averaging
    # them with a simple running sum keeps the result wide — no unstack
    # needed (the earlier .mean(axis=1) bug collapsed columns instead).
    rank_cols = [f"rank_{c}" for c in
                 ["pe_ratio", "pb_ratio", "ev_ebitda", "earnings_yield", "dividend_yield"]
                 if f"rank_{c}" in feat]
    if len(rank_cols) >= 2:
        acc = feat[rank_cols[0]].copy()
        for c in rank_cols[1:]:
            acc = acc.add(feat[c], fill_value=0)
        feat["value_composite"] = acc / len(rank_cols)

    # EY + DY composite
    if "earnings_yield" in fun.columns and "dividend_yield" in fun.columns:
        ey = fun["earnings_yield"].unstack("ticker")
        dy = fun["dividend_yield"].unstack("ticker")
        feat["yield_composite"] = (ey + dy) / 2

    # Piotroski-style cheap + profitable
    if "earnings_yield" in fun.columns and "roe" in fun.columns:
        ey  = fun["earnings_yield"].unstack("ticker")
        roe = fun["roe"].unstack("ticker")
        feat["value_quality_combo"] = (
            ey.rank(axis=1, pct=True) + roe.rank(axis=1, pct=True)
        ) / 2

    # ── 7. Debt-adjusted earnings yield (2 features) ─────────────────────────
    if "earnings_yield" in fun.columns and "debt_equity" in fun.columns:
        ey = fun["earnings_yield"].unstack("ticker")
        de = fun["debt_equity"].unstack("ticker")
        feat["debt_adj_ey"] = ey / (1 + de.clip(lower=0))
        feat["net_ey"]      = ey - 0.05 * de.clip(lower=0, upper=3)

    # ── 8. EV/EBITDA-based discount (3 features) ─────────────────────────────
    if "ev_ebitda" in fun.columns:
        ev = fun["ev_ebitda"].unstack("ticker")
        feat["ev_ebitda_rank"]   = (1 - ev.rank(axis=1, pct=True))
        feat["ev_ebitda_zscore"] = feat["zscore_ev_ebitda"] if "zscore_ev_ebitda" in feat else ev.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)
        feat["low_ev_flag"]      = (ev < ev.quantile(0.3, axis=1).values[:, None]).astype(float)

    # ── 9. PB-ROE relationship (2 features) ──────────────────────────────────
    if "pb_ratio" in fun.columns and "roe" in fun.columns:
        pb  = fun["pb_ratio"].unstack("ticker")
        roe = fun["roe"].unstack("ticker")
        feat["pb_roe_spread"] = roe - 0.1 * pb          # higher = undervalued given ROE
        feat["roe_pb_ratio"]  = roe / (pb.replace(0, np.nan))

    # ── 10. Current ratio as value quality gate (2 features) ─────────────────
    if "current_ratio" in fun.columns:
        cr = fun["current_ratio"].unstack("ticker")
        feat["current_ratio_rank"]   = cr.rank(axis=1, pct=True)
        feat["high_current_ratio"]   = (cr > 1.5).astype(float)

    # ── 11. Dividend yield trend and quality (3 features) ────────────────────
    if "dividend_yield" in fun.columns:
        dy = fun["dividend_yield"].unstack("ticker")
        feat["div_yield_change_63d"]  = dy.diff(63)
        feat["div_yield_change_252d"] = dy.diff(252)
        feat["div_yield_zscore"] = feat.get(
            "zscore_dividend_yield",
            dy.sub(dy.mean(axis=1), axis=0).div(dy.std(axis=1).replace(0, np.nan), axis=0),
        )

    # ── 12. Earnings yield momentum (2 features) ─────────────────────────────
    if "earnings_yield" in fun.columns:
        ey = fun["earnings_yield"].unstack("ticker")
        feat["ey_change_63d"]  = ey.diff(63)
        feat["ey_accel"]       = ey.diff(63) - ey.diff(126)

    # ── 13. Composite "cheapness vs history" — own time-series rank (3) ─────
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
        if col in fun.columns:
            raw = fun[col].unstack("ticker")
            ts_rank = raw.rank(axis=0, pct=True)
            feat[f"ts_rank_{col}"] = (1 - ts_rank)  # cheap relative to own history = high score

    # ── 14. Debt-to-earnings adjusted multiples (2 features) ─────────────────
    if "ev_ebitda" in fun.columns and "debt_equity" in fun.columns:
        ev = fun["ev_ebitda"].unstack("ticker")
        de = fun["debt_equity"].unstack("ticker")
        feat["leverage_adj_ev_ebitda"] = ev * (1 + de.clip(lower=0))
        feat["leverage_adj_ev_rank"]   = (1 - feat["leverage_adj_ev_ebitda"].rank(axis=1, pct=True))

    # ── Stack and return ──────────────────────────────────────────────────────
    # Some feats may already be stacked Series; normalise to wide then stack
    wide = {}
    for k, v in feat.items():
        if isinstance(v, pd.DataFrame):
            wide[k] = v
        elif isinstance(v, pd.Series) and isinstance(v.index, pd.MultiIndex):
            wide[k] = v.unstack("ticker")
        else:
            wide[k] = v

    result = pd.concat(
        {k: v.stack() for k, v in wide.items()},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
