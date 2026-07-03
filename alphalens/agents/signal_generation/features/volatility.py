"""
volatility.py
Volatility Features — AlphaLens Signal Generation Agent
~40 features capturing risk, realized volatility, and beta.

Features:
  - Historical volatility (multiple windows)
  - Volatility-of-volatility
  - Average True Range (ATR)
  - Downside deviation / semi-deviation
  - Beta (rolling market beta)
  - Idiosyncratic volatility
  - Volatility regime signals
  - Volatility ratios
  - Max drawdown (rolling)
  - VaR / CVaR proxies
"""

import pandas as pd
import numpy as np


def compute_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility features.

    Args:
        prices: MultiIndex (date, ticker) with adj_close, high, low, returns.

    Returns:
        DataFrame with volatility feature columns.
    """
    close  = prices["adj_close"].unstack("ticker")
    ret    = prices["returns"].unstack("ticker")

    # Market returns (equal-weight cross-sectional mean)
    mkt_ret = ret.mean(axis=1)

    feat: dict = {}

    # ── 1. Historical volatility (6 features) ────────────────────────────────
    for n in [5, 10, 20, 60, 120, 252]:
        feat[f"vol_{n}d"] = ret.rolling(n, min_periods=n // 2).std() * np.sqrt(252)

    # ── 2. Volatility-of-volatility (3 features) ─────────────────────────────
    vol21  = ret.rolling(21, min_periods=10).std()
    for n in [20, 60, 120]:
        feat[f"vol_of_vol_{n}d"] = vol21.rolling(n, min_periods=n // 2).std() * np.sqrt(252)

    # ── 3. Average True Range (3 features) ───────────────────────────────────
    if "high" in prices.columns and "low" in prices.columns:
        high = prices["high"].unstack("ticker")
        low  = prices["low"].unstack("ticker")
        tr   = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=0,
        ).groupby(level=0).max() if False else (high - low).combine(
            (high - close.shift(1)).abs(), np.maximum
        ).combine(
            (low - close.shift(1)).abs(), np.maximum
        )
        for n in [7, 14, 21]:
            feat[f"atr_{n}d"] = tr.rolling(n, min_periods=n // 2).mean() / (close + 1e-8)
    else:
        # Proxy ATR from close
        daily_range = (close.pct_change().abs())
        for n in [7, 14, 21]:
            feat[f"atr_{n}d"] = daily_range.rolling(n, min_periods=n // 2).mean()

    # ── 4. Downside deviation (3 features) ───────────────────────────────────
    for n in [20, 60, 252]:
        down = ret.copy()
        down[down > 0] = 0
        feat[f"downside_dev_{n}d"] = down.rolling(n, min_periods=n // 2).std() * np.sqrt(252)

    # ── 5. Semi-deviation ratio (Sortino proxy) (3 features) ─────────────────
    for n in [20, 60, 252]:
        total_vol  = ret.rolling(n, min_periods=n // 2).std()
        down       = ret.copy(); down[down > 0] = 0
        down_vol   = down.rolling(n, min_periods=n // 2).std()
        feat[f"downside_ratio_{n}d"] = down_vol / (total_vol + 1e-8)

    # ── 6. Rolling beta to equal-weight market (3 features) ──────────────────
    for n in [60, 120, 252]:
        mkt_var = mkt_ret.rolling(n, min_periods=n // 2).var()
        cov_    = ret.apply(
            lambda col: col.rolling(n, min_periods=n // 2).cov(mkt_ret)
        )
        feat[f"beta_{n}d"] = cov_.div(mkt_var + 1e-10, axis=0)

    # ── 7. Idiosyncratic volatility (2 features) ─────────────────────────────
    for n in [60, 252]:
        beta_n    = feat[f"beta_{n}d"]
        mkt_vol   = mkt_ret.rolling(n, min_periods=n // 2).std() * np.sqrt(252)
        sys_vol   = beta_n.multiply(mkt_vol, axis=0)
        total_vol = feat[f"vol_{n}d"]
        idio_var  = (total_vol ** 2 - sys_vol ** 2).clip(lower=0)
        feat[f"idio_vol_{n}d"] = idio_var.apply(np.sqrt)

    # ── 8. Volatility regime (high/low vol flag) (2 features) ────────────────
    vol21_series = feat["vol_20d"]
    roll_med_vol = vol21_series.rolling(252, min_periods=126).median()
    feat["high_vol_flag"] = (vol21_series > roll_med_vol).astype(float)
    feat["low_vol_flag"]  = (vol21_series < roll_med_vol).astype(float)

    # ── 9. Volatility ratio: short-term vs long-term (3 features) ────────────
    feat["vol_ratio_5_20"]   = feat["vol_5d"]  / (feat["vol_20d"]  + 1e-8)
    feat["vol_ratio_20_60"]  = feat["vol_20d"] / (feat["vol_60d"]  + 1e-8)
    feat["vol_ratio_60_252"] = feat["vol_60d"] / (feat["vol_252d"] + 1e-8)

    # ── 10. Rolling max drawdown (3 features) ────────────────────────────────
    for n in [20, 60, 252]:
        wealth    = (1 + ret.fillna(0)).cumprod()
        roll_peak = wealth.rolling(n, min_periods=1).max()
        dd        = (wealth - roll_peak) / (roll_peak + 1e-8)
        feat[f"max_dd_{n}d"] = dd.rolling(n, min_periods=1).min()

    # ── 11. VaR proxy (historical) (2 features) ──────────────────────────────
    feat["var_5pct_20d"]  = ret.rolling(20,  min_periods=10).quantile(0.05)
    feat["var_5pct_252d"] = ret.rolling(252, min_periods=126).quantile(0.05)

    # ── 12. Skewness and kurtosis of returns (4 features) ────────────────────
    for n in [20, 60]:
        feat[f"skew_{n}d"] = ret.rolling(n, min_periods=n // 2).skew()
        feat[f"kurt_{n}d"] = ret.rolling(n, min_periods=n // 2).kurt()

    # ── 13. Volatility rank (cross-sectional) (2 features) ───────────────────
    feat["vol_rank_20d"]  = feat["vol_20d"].rank(axis=1, pct=True)
    feat["vol_rank_60d"]  = feat["vol_60d"].rank(axis=1, pct=True)

    # ── 14. Parkinson range-based volatility estimator (2 features) ──────────
    if "high" in prices.columns and "low" in prices.columns:
        high = prices["high"].unstack("ticker")
        low  = prices["low"].unstack("ticker")
        log_hl_sq = (np.log(high / (low + 1e-8))) ** 2
        for n in [20, 60]:
            feat[f"parkinson_vol_{n}d"] = np.sqrt(
                log_hl_sq.rolling(n, min_periods=n // 2).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

    # ── Stack and return ──────────────────────────────────────────────────────
    result = pd.concat(
        {k: v.stack() for k, v in feat.items()},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
