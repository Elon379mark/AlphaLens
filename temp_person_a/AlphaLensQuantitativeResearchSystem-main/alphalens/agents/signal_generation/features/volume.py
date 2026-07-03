"""
volume.py
Volume Features — AlphaLens Signal Generation Agent
~35 features capturing liquidity, turnover, and volume-price relationships.

Features:
  - VWAP deviation
  - On-Balance Volume (OBV) momentum
  - Amihud illiquidity ratio
  - Volume turnover ratios
  - Volume-price trend
  - Abnormal volume
  - Dollar volume
  - Volume momentum (volume vs its own MA)
  - Bid-ask spread proxy (high-low / close)
  - Kyle's lambda proxy
"""

import pandas as pd
import numpy as np


def compute_volume_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume features.

    Args:
        prices: MultiIndex (date, ticker) with adj_close, volume, returns.

    Returns:
        DataFrame with volume feature columns.
    """
    close  = prices["adj_close"].unstack("ticker")
    vol    = prices["volume"].unstack("ticker")
    ret    = prices["returns"].unstack("ticker")

    feat: dict = {}

    # ── 1. VWAP and VWAP deviation (3 features) ──────────────────────────────
    pv = close * vol
    for n in [5, 20, 60]:
        vwap = pv.rolling(n, min_periods=n // 2).sum() / (vol.rolling(n, min_periods=n // 2).sum() + 1e-8)
        feat[f"vwap_dev_{n}d"] = (close / (vwap + 1e-8)) - 1

    # ── 2. On-Balance Volume momentum (3 features) ───────────────────────────
    direction = ret.apply(lambda x: x.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0)))
    obv       = (direction * vol).cumsum()
    for n in [10, 20, 60]:
        feat[f"obv_mom_{n}d"] = (obv / (obv.shift(n) + 1e-8)) - 1

    # ── 3. Amihud illiquidity ratio (3 features) ─────────────────────────────
    dv = (close * vol).replace(0, np.nan)
    amihud_daily = ret.abs() / dv
    for n in [20, 60, 252]:
        feat[f"amihud_{n}d"] = amihud_daily.rolling(n, min_periods=n // 2).mean() * 1e6

    # ── 4. Abnormal volume (3 features) ──────────────────────────────────────
    for n in [5, 20, 60]:
        vol_ma = vol.rolling(n, min_periods=n // 2).mean()
        feat[f"abnorm_vol_{n}d"] = vol / (vol_ma + 1e-8)

    # ── 5. Dollar volume (3 features) ────────────────────────────────────────
    dollar_vol = close * vol
    for n in [20, 60, 252]:
        feat[f"dollar_vol_{n}d"] = dollar_vol.rolling(n, min_periods=n // 2).mean().apply(np.log1p)

    # ── 6. Volume momentum (volume vs own MA) (3 features) ───────────────────
    for n in [5, 10, 20]:
        vol_ma = vol.rolling(n * 3, min_periods=n).mean()
        feat[f"vol_ratio_{n}d"] = vol.rolling(n).mean() / (vol_ma + 1e-8)

    # ── 7. Volume-Price Trend (VPT) momentum (2 features) ────────────────────
    vpt = (ret * vol).cumsum()
    feat["vpt_mom_20d"]  = (vpt / (vpt.shift(20)  + 1e-8)) - 1
    feat["vpt_mom_60d"]  = (vpt / (vpt.shift(60)  + 1e-8)) - 1

    # ── 8. Turnover (volume / shares proxy) (3 features) ─────────────────────
    # Proxy: normalize volume by its rolling max
    vol_max = vol.rolling(252, min_periods=63).max().replace(0, np.nan)
    feat["turnover_norm"]    = vol / vol_max
    feat["turnover_ma_20d"]  = feat["turnover_norm"].rolling(20, min_periods=10).mean()
    feat["high_turnover"]    = (feat["turnover_norm"] > feat["turnover_norm"].rolling(252, min_periods=63).quantile(0.8)).astype(float)

    # ── 9. Bid-ask spread proxy: (high - low) / close (3 features) ───────────
    if "high" in prices.columns and "low" in prices.columns:
        high = prices["high"].unstack("ticker")
        low  = prices["low"].unstack("ticker")
        hl_spread = (high - low) / (close + 1e-8)
        feat["spread_proxy_1d"]   = hl_spread
        feat["spread_proxy_20d"]  = hl_spread.rolling(20, min_periods=10).mean()
        feat["spread_proxy_60d"]  = hl_spread.rolling(60, min_periods=30).mean()
    else:
        # Proxy from return volatility
        feat["spread_proxy_1d"]  = ret.abs()
        feat["spread_proxy_20d"] = ret.abs().rolling(20, min_periods=10).mean()
        feat["spread_proxy_60d"] = ret.abs().rolling(60, min_periods=30).mean()

    # ── 10. Kyle's lambda proxy (price impact) (3 features) ──────────────────
    signed_vol = vol * ret.apply(np.sign)
    for n in [20, 60, 252]:
        # Regression coefficient of |ret| on signed_vol
        abs_ret    = ret.abs().rolling(n, min_periods=n // 2)
        signed_vol_roll = signed_vol.rolling(n, min_periods=n // 2)
        # Approximate lambda as ratio of mean |ret| / mean |vol|
        feat[f"kyle_lambda_{n}d"] = (
            ret.abs().rolling(n, min_periods=n // 2).mean() /
            (vol.rolling(n, min_periods=n // 2).mean() + 1e-8)
        ) * 1e6

    # ── 11. Relative volume rank (2 features) ────────────────────────────────
    feat["vol_rank_20d"]  = vol.rolling(20, min_periods=10).mean().rank(axis=1, pct=True)
    feat["vol_rank_252d"] = vol.rolling(252, min_periods=126).mean().rank(axis=1, pct=True)

    # ── 12. Volume acceleration and trend (2 features) ───────────────────────
    feat["vol_accel_5_20"] = (
        vol.rolling(5, min_periods=3).mean() / (vol.rolling(20, min_periods=10).mean() + 1e-8) - 1
    )
    feat["vol_trend_60d"] = vol.rolling(60, min_periods=30).mean().pct_change(20)

    # ── 13. Price-volume correlation (1 feature) ──────────────────────────────
    feat["price_vol_corr_60d"] = ret.rolling(60, min_periods=30).corr(vol.pct_change())

    # ── Stack and return ──────────────────────────────────────────────────────
    result = pd.concat(
        {k: (v.stack() if isinstance(v, pd.DataFrame) else v) for k, v in feat.items()},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
