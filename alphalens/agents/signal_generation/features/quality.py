"""
quality.py
Quality Features — AlphaLens Signal Generation Agent
~45 features capturing business quality and earnings reliability.

Features:
  - Profitability: ROE, ROA, gross margin, net margin
  - Accruals (Sloan 1996)
  - Asset growth (Cooper et al. 2008)
  - Earnings stability
  - Piotroski F-Score components
  - Leverage / financial health
  - Cross-sectional ranks and z-scores
  - Quality composites
"""

import pandas as pd
import numpy as np


def compute_quality_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality features from fundamental data.

    Args:
        fundamentals: MultiIndex (date, ticker) DataFrame.

    Returns:
        DataFrame with quality feature columns.
    """
    fun = fundamentals.unstack("ticker")
    feat: dict = {}

    def _get(col: str) -> pd.DataFrame | None:
        return fun[col] if col in fun.columns else None

    roe  = _get("roe")
    roa  = _get("roa")
    gm   = _get("gross_margin")
    acc  = _get("accruals")
    ag   = _get("asset_growth")
    de   = _get("debt_equity")
    cr   = _get("current_ratio")
    ey   = _get("earnings_yield")
    dy   = _get("dividend_yield")

    # ── 1. Raw profitability (5 features) ────────────────────────────────────
    for name, s in [("roe", roe), ("roa", roa), ("gross_margin", gm),
                    ("earnings_yield", ey), ("dividend_payout", dy)]:
        if s is not None:
            feat[f"qual_{name}"] = s

    # ── 2. Cross-sectional ranks (5 features) ────────────────────────────────
    for name, s in [("roe", roe), ("roa", roa), ("gross_margin", gm)]:
        if s is not None:
            feat[f"rank_{name}"] = s.rank(axis=1, pct=True)

    # ── 3. Z-scores (5 features) ─────────────────────────────────────────────
    for name, s in [("roe", roe), ("roa", roa), ("gross_margin", gm),
                    ("accruals", acc), ("asset_growth", ag)]:
        if s is not None:
            cs_mean = s.mean(axis=1)
            cs_std  = s.std(axis=1).replace(0, np.nan)
            feat[f"zscore_{name}"] = s.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # ── 4. Accruals (Sloan) — lower is better (3 features) ───────────────────
    if acc is not None:
        feat["accruals_rank"]   = (1 - acc.rank(axis=1, pct=True))  # low acc = high rank
        feat["low_accruals"]    = (acc < acc.quantile(0.3, axis=1).values[:, None]).astype(float)
        feat["accruals_change"] = acc.pct_change(4)          # quarterly change

    # ── 5. Asset growth (Cooper) — lower is better (3 features) ──────────────
    if ag is not None:
        feat["asset_growth_rank"]   = (1 - ag.rank(axis=1, pct=True))
        feat["low_asset_growth"]    = (ag < ag.quantile(0.3, axis=1).values[:, None]).astype(float)
        feat["asset_growth_change"] = ag.diff(4)

    # ── 6. Piotroski F-Score components (9 features) ─────────────────────────
    # F1: ROA > 0
    if roa is not None:
        feat["f_roa_positive"]   = (roa > 0).astype(float)
        feat["f_roa_improving"]  = (roa.diff(4) > 0).astype(float)

    # F2: Operating cash flow > 0 (proxy: ROA + accruals)
    if roa is not None and acc is not None:
        feat["f_cfo_positive"]  = ((roa - acc) > 0).astype(float)

    # F3: Accruals < 0 (cash earnings > accruals)
    if acc is not None:
        feat["f_accruals_low"]  = (acc < 0).astype(float)

    # F4: Leverage declining
    if de is not None:
        feat["f_leverage_low"]  = (de.diff(4) < 0).astype(float)
        feat["f_low_leverage"]  = (de < de.quantile(0.4, axis=1).values[:, None]).astype(float)

    # F5: Current ratio improving
    if cr is not None:
        feat["f_cr_improving"]  = (cr.diff(4) > 0).astype(float)
        feat["f_high_cr"]       = (cr > 1.0).astype(float)

    # F6: Gross margin improving
    if gm is not None:
        feat["f_gm_improving"]  = (gm.diff(4) > 0).astype(float)

    # Composite F-Score
    f_cols = [c for c in feat if c.startswith("f_")]
    if len(f_cols) >= 4:
        feat["piotroski_f"] = pd.concat([feat[c] for c in f_cols], axis=1).sum(axis=1).unstack() \
            if not isinstance(feat[f_cols[0]], pd.DataFrame) \
            else pd.concat([feat[c] for c in f_cols], axis=0).groupby(level=0).sum()

    # ── 7. Profitability trends (4 features) ─────────────────────────────────
    if roe is not None:
        feat["roe_trend_4q"]  = roe.diff(4)
        feat["roe_trend_8q"]  = roe.diff(8)
    if gm is not None:
        feat["gm_trend_4q"]   = gm.diff(4)
        feat["gm_trend_8q"]   = gm.diff(8)

    # ── 8. Composite quality scores (3 features) ─────────────────────────────
    rank_components = [feat[c] for c in ["rank_roe", "rank_roa", "rank_gross_margin"]
                       if c in feat]
    if len(rank_components) >= 2:
        feat["quality_composite"] = sum(rank_components) / len(rank_components)

    if roe is not None and acc is not None:
        feat["quality_accrual_adj"] = (
            roe.rank(axis=1, pct=True) +
            (1 - acc.rank(axis=1, pct=True))
        ) / 2

    if roe is not None and de is not None:
        feat["quality_leverage_adj"] = roe / (1 + de.clip(lower=0))

    # ── 9. Earnings yield stability (3 features) ─────────────────────────────
    if ey is not None:
        feat["ey_std_4q"]    = ey.rolling(4, min_periods=2).std()
        feat["ey_cv"]        = ey.rolling(4, min_periods=2).std() / (ey.rolling(4).mean().abs() + 1e-8)
        feat["ey_improving"] = (ey.diff(4) > 0).astype(float)

    # ── 10. ROA/ROE trend and stability (4 features) ──────────────────────────
    if roa is not None:
        feat["roa_trend_4q"] = roa.diff(4)
        feat["roa_std_4q"]   = roa.rolling(4, min_periods=2).std()
    if roe is not None:
        feat["roe_std_4q"]   = roe.rolling(4, min_periods=2).std()
        feat["roe_cv"]       = roe.rolling(4, min_periods=2).std() / (roe.rolling(4).mean().abs() + 1e-8)

    # ── 11. Combined leverage-quality gate (3 features) ───────────────────────
    if de is not None and cr is not None:
        feat["leverage_liquidity_score"] = cr.rank(axis=1, pct=True) - de.rank(axis=1, pct=True)
    if de is not None and roe is not None:
        feat["financial_health_score"]   = roe.rank(axis=1, pct=True) - de.rank(axis=1, pct=True)
    if de is not None:
        feat["deleveraging_flag"] = (de.diff(8) < 0).astype(float)

    # ── 12. Gross margin level vs history (own time-series rank) (2 features) ─
    if gm is not None:
        feat["gm_ts_rank"]   = gm.rank(axis=0, pct=True)
        feat["gm_above_avg"] = (gm > gm.rolling(8, min_periods=4).mean()).astype(float)

    # ── 13. Quality momentum: change in quality_composite (1 feature) ────────
    if "quality_composite" in feat:
        feat["quality_composite_change"] = feat["quality_composite"].diff(4)

    # ── Stack and return ──────────────────────────────────────────────────────
    wide = {}
    for k, v in feat.items():
        if isinstance(v, pd.DataFrame):
            wide[k] = v
        elif isinstance(v, pd.Series) and isinstance(v.index, pd.MultiIndex):
            wide[k] = v.unstack("ticker")
        else:
            wide[k] = v

    result = pd.concat(
        {k: v.stack() for k, v in wide.items() if isinstance(v, pd.DataFrame)},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
