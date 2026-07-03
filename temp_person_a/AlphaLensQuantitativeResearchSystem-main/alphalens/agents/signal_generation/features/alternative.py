"""
alternative.py
Alternative Data Features — AlphaLens Signal Generation Agent
~30 features: short interest, options implied volatility, analyst revisions,
and other non-traditional signal sources.

Note: In production these would be sourced from vendor feeds (e.g. ORATS for
options, S3 Partners for short interest, IBES for analyst data). Here, since
fundamentals already carry proxy columns, we derive alternative signals from
available data plus simple synthetic proxies clearly marked as such. Replace
the `_proxy` suffixed functions with real vendor data when available.
"""

import pandas as pd
import numpy as np


def compute_alternative_features(
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute alternative-data features.

    Args:
        ohlcv: MultiIndex (date, ticker) with adj_close, volume, returns.
        fundamentals: MultiIndex (date, ticker) with fundamental columns.

    Returns:
        DataFrame with alternative feature columns.
    """
    close = ohlcv["adj_close"].unstack("ticker")
    vol   = ohlcv["volume"].unstack("ticker")
    ret   = ohlcv["returns"].unstack("ticker")

    # Align fundamentals to OHLCV dates via forward-fill (union-then-ffill,
    # see value.py for why naive reindex-before-ffill loses all observations).
    fun_wide = fundamentals.unstack("ticker")
    union_dates = fun_wide.index.union(close.index).sort_values()
    fun = (
        fun_wide
        .reindex(union_dates)
        .ffill()
        .reindex(close.index)
    )

    feat: dict = {}

    # ── 1. Short-interest proxy: high realized vol + negative momentum ───────
    # True short interest needs vendor data; we proxy "crowded short" risk via
    # the interaction of high volatility and negative momentum, which
    # correlates with elevated short interest in practice.
    vol20 = ret.rolling(20, min_periods=10).std()
    mom20 = close.pct_change(20)
    feat["short_interest_proxy"] = vol20.rank(axis=1, pct=True) * (1 - mom20.rank(axis=1, pct=True))
    feat["days_to_cover_proxy"]  = vol20 / (vol.rolling(20, min_periods=10).mean().apply(np.log1p) + 1e-8)
    feat["short_squeeze_risk"]   = (mom20 > mom20.quantile(0.8, axis=1).values[:, None]).astype(float) * \
                                    (vol20 > vol20.quantile(0.7, axis=1).values[:, None]).astype(float)

    # ── 2. Options-implied volatility proxy ──────────────────────────────────
    # True IV requires options chain data; proxy via realized vol term structure
    rv_10  = ret.rolling(10, min_periods=5).std() * np.sqrt(252)
    rv_60  = ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    feat["iv_proxy_short"]       = rv_10
    feat["iv_term_structure"]    = rv_10 / (rv_60 + 1e-8)         # >1 = backwardation (stress)
    feat["iv_skew_proxy"]        = ret.rolling(60, min_periods=30).skew()
    feat["iv_kurtosis_proxy"]    = ret.rolling(60, min_periods=30).kurt()
    feat["put_call_skew_proxy"]  = -ret.rolling(20, min_periods=10).skew()  # neg skew → put demand

    # ── 3. Analyst revision proxy via earnings yield momentum ────────────────
    # True analyst revisions need IBES estimate data; proxy via the rate of
    # change in earnings yield, which tracks fundamental re-rating direction.
    # Uses the forward-filled daily series (fun) rather than raw quarterly
    # fundamentals, since diff() on sparse quarterly data is mostly NaN.
    if "earnings_yield" in fun.columns:
        ey = fun["earnings_yield"]
        feat["analyst_revision_proxy_1q"] = ey.diff(63)
        feat["analyst_revision_proxy_2q"] = ey.diff(126)
        feat["revision_rank"]             = ey.diff(63).rank(axis=1, pct=True)
        feat["revision_momentum"]         = ey.diff(63) - ey.diff(126)
        feat["revision_acceleration"]     = ey.diff(21) - ey.diff(42)

    # ── 4. Institutional flow proxy via abnormal dollar volume ──────────────
    dollar_vol = close * vol
    dv_ma60    = dollar_vol.rolling(60, min_periods=30).mean()
    feat["inst_flow_proxy"]       = (dollar_vol.rolling(5, min_periods=3).mean() / (dv_ma60 + 1e-8)) - 1
    feat["smart_money_proxy"]     = (ret.rolling(5, min_periods=3).mean() *
                                      (dollar_vol.rolling(5, min_periods=3).mean() / (dv_ma60 + 1e-8)))

    # ── 5. Sentiment proxy via price acceleration + volume confirmation ─────
    accel = close.pct_change(10) - close.pct_change(20)
    vol_confirm = (vol.rolling(10, min_periods=5).mean() / (vol.rolling(60, min_periods=30).mean() + 1e-8))
    feat["sentiment_proxy"]      = accel * vol_confirm
    feat["news_momentum_proxy"]  = ret.rolling(3, min_periods=2).sum() * vol_confirm

    # ── 6. Seasonality / calendar effects (5 features) ───────────────────────
    dates = close.index
    month = pd.Series(dates.month, index=dates)
    dow   = pd.Series(dates.dayofweek, index=dates)
    day_of_month = pd.Series(dates.day, index=dates)

    feat["is_january"]      = pd.DataFrame(
        np.tile((month == 1).astype(float).values[:, None], (1, close.shape[1])),
        index=close.index, columns=close.columns,
    )
    feat["is_month_end"]    = pd.DataFrame(
        np.tile((day_of_month >= 25).astype(float).values[:, None], (1, close.shape[1])),
        index=close.index, columns=close.columns,
    )
    feat["is_monday"]       = pd.DataFrame(
        np.tile((dow == 0).astype(float).values[:, None], (1, close.shape[1])),
        index=close.index, columns=close.columns,
    )
    feat["is_friday"]       = pd.DataFrame(
        np.tile((dow == 4).astype(float).values[:, None], (1, close.shape[1])),
        index=close.index, columns=close.columns,
    )

    # ── 7. Macro-sensitivity proxy via cross-sectional dispersion ───────────
    cs_dispersion = ret.std(axis=1)
    feat["macro_regime_proxy"] = pd.DataFrame(
        np.tile(cs_dispersion.values[:, None], (1, close.shape[1])),
        index=close.index, columns=close.columns,
    )

    # ── 8. Supply-chain / sector momentum spillover proxy ───────────────────
    if "sector" in fun.columns:
        sector = fun["sector"]
        sector_ret = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
        for col in ret.columns:
            sec = sector[col].mode().iloc[0] if col in sector.columns and not sector[col].dropna().empty else None
            if sec is not None:
                peer_cols = [c for c in ret.columns if c in sector.columns and
                             (sector[c] == sec).any()]
                if len(peer_cols) > 1:
                    sector_ret[col] = ret[peer_cols].drop(columns=[col], errors="ignore").mean(axis=1)
        feat["sector_spillover_mom"] = sector_ret.rolling(20, min_periods=10).mean()

    # ── 9. Liquidity stress proxy (combination signal) (2 features) ─────────
    feat["liquidity_stress_proxy"] = (
        feat["iv_term_structure"].rank(axis=1, pct=True) +
        feat["short_interest_proxy"].rank(axis=1, pct=True)
    ) / 2
    feat["flight_to_quality_proxy"] = -feat["sentiment_proxy"] * feat["macro_regime_proxy"]

    # ── 10. Crowding score (positioning proxy) (2 features) ──────────────────
    feat["crowding_score"] = (
        feat["short_interest_proxy"].rank(axis=1, pct=True) +
        feat["inst_flow_proxy"].rank(axis=1, pct=True)
    ) / 2
    feat["contrarian_signal"] = -feat["crowding_score"] * np.sign(close.pct_change(20))

    # ── 11. Volatility risk premium proxy (1 feature) ─────────────────────────
    feat["vol_risk_premium_proxy"] = feat["iv_proxy_short"] - feat["vol_of_vol_proxy"] if "vol_of_vol_proxy" in feat else feat["iv_proxy_short"] - rv_60

    # ── Stack and return ──────────────────────────────────────────────────────
    result = pd.concat(
        {k: v.stack() for k, v in feat.items() if isinstance(v, pd.DataFrame)},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
