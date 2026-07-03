"""
momentum.py
Momentum Features — AlphaLens Signal Generation Agent
~60 features capturing price momentum across multiple horizons.

Features:
  - Raw return momentum (6 windows)
  - Skip-month momentum (12-1, 6-1, 3-1)
  - MACD variants (standard + fast + slow)
  - Momentum acceleration
  - Residual momentum
  - Momentum dispersion
  - Rate of change (ROC) variants
  - Relative strength variants
  - Reversal signals (short-term)
  - Momentum consistency (% positive months)
  - Drawdown from peak
  - 52-week high proximity
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all momentum features.

    Args:
        prices: MultiIndex (date, ticker) DataFrame with 'adj_close' and 'returns'.

    Returns:
        DataFrame with momentum feature columns, same index as prices.
    """
    close = prices["adj_close"].unstack("ticker")
    ret = prices["returns"].unstack("ticker")

    feat: Dict[str, pd.DataFrame] = {}

    # ── 1. Raw return momentum (6 features) ─────────────────────────────────
    for n in [5, 10, 20, 60, 120, 252]:
        feat[f"mom_{n}d"] = close.pct_change(n)

    # ── 2. Skip-month momentum (3 features) ─────────────────────────────────
    feat["mom_12_1"] = close.pct_change(252) - close.pct_change(21)
    feat["mom_6_1"]  = close.pct_change(126) - close.pct_change(21)
    feat["mom_3_1"]  = close.pct_change(63)  - close.pct_change(21)

    # ── 3. MACD variants (6 features) ────────────────────────────────────────
    # Standard (12, 26, 9)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"]  = macd - signal
    feat["macd_cross"] = (
        (macd > signal).astype(int) - (macd.shift(1) > signal.shift(1)).astype(int)
    )
    feat["macd_ratio"] = macd / (close.abs() + 1e-8)

    # Fast (5, 13, 4)
    ema5   = close.ewm(span=5, adjust=False).mean()
    ema13  = close.ewm(span=13, adjust=False).mean()
    macd_f = ema5 - ema13
    sig_f  = macd_f.ewm(span=4, adjust=False).mean()
    feat["macd_fast_hist"]  = macd_f - sig_f
    feat["macd_fast_cross"] = (
        (macd_f > sig_f).astype(int) - (macd_f.shift(1) > sig_f.shift(1)).astype(int)
    )

    # Slow (19, 39, 9)
    ema19  = close.ewm(span=19, adjust=False).mean()
    ema39  = close.ewm(span=39, adjust=False).mean()
    macd_s = ema19 - ema39
    sig_s  = macd_s.ewm(span=9, adjust=False).mean()
    feat["macd_slow_hist"] = macd_s - sig_s

    # ── 4. Momentum acceleration (4 features) ────────────────────────────────
    feat["mom_accel_20_40"]   = close.pct_change(20) - close.pct_change(40)
    feat["mom_accel_60_120"]  = close.pct_change(60) - close.pct_change(120)
    feat["mom_accel_5_20"]    = close.pct_change(5)  - close.pct_change(20)
    feat["mom_accel_20_60"]   = close.pct_change(20) - close.pct_change(60)

    # ── 5. Residual momentum (2 features) ────────────────────────────────────
    feat["residual_mom"]        = close.pct_change(252) - close.pct_change(21)
    feat["residual_mom_6_1"]    = close.pct_change(126) - close.pct_change(21)

    # ── 6. Rate of Change variants (5 features) ──────────────────────────────
    for n in [3, 7, 14, 21, 42]:
        feat[f"roc_{n}d"] = (close / close.shift(n) - 1)

    # ── 7. 52-week high proximity (2 features) ───────────────────────────────
    roll_max_252 = close.rolling(252, min_periods=63).max()
    roll_min_252 = close.rolling(252, min_periods=63).min()
    feat["dist_52w_high"] = (close / (roll_max_252 + 1e-8)) - 1
    feat["dist_52w_low"]  = (close / (roll_min_252 + 1e-8)) - 1

    # ── 8. Momentum consistency (3 features) ─────────────────────────────────
    # Fraction of positive days in past N periods
    pos = (ret > 0).astype(float)
    feat["mom_consist_20"]  = pos.rolling(20,  min_periods=10).mean()
    feat["mom_consist_60"]  = pos.rolling(60,  min_periods=30).mean()
    feat["mom_consist_252"] = pos.rolling(252, min_periods=126).mean()

    # ── 9. Drawdown from rolling peak (3 features) ───────────────────────────
    for n in [20, 60, 252]:
        roll_peak = close.rolling(n, min_periods=n // 2).max()
        feat[f"drawdown_{n}d"] = (close - roll_peak) / (roll_peak + 1e-8)

    # ── 10. Relative strength vs cross-sectional mean (3 features) ───────────
    for n in [20, 60, 252]:
        cs_mean = close.pct_change(n).mean(axis=1)
        feat[f"rel_strength_{n}d"] = close.pct_change(n).sub(cs_mean, axis=0)

    # ── 11. Short-term reversal (3 features) ─────────────────────────────────
    feat["reversal_1w"]  = -close.pct_change(5)
    feat["reversal_2w"]  = -close.pct_change(10)
    feat["reversal_1m"]  = -close.pct_change(21)

    # ── 12. EMA ratio (price / EMA) — trend signals (5 features) ────────────
    for span in [10, 20, 50, 100, 200]:
        ema = close.ewm(span=span, adjust=False).mean()
        feat[f"price_ema_ratio_{span}"] = close / (ema + 1e-8) - 1

    # ── 13. SMA cross signals (4 features) ───────────────────────────────────
    sma10  = close.rolling(10).mean()
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["sma_cross_10_20"]   = (sma10  / (sma20  + 1e-8)) - 1
    feat["sma_cross_20_50"]   = (sma20  / (sma50  + 1e-8)) - 1
    feat["sma_cross_50_200"]  = (sma50  / (sma200 + 1e-8)) - 1
    feat["sma_cross_10_50"]   = (sma10  / (sma50  + 1e-8)) - 1

    # ── 14. Momentum z-score (cross-sectional standardisation) (3 features) ──
    for n in [20, 60, 252]:
        raw = close.pct_change(n)
        cs_std  = raw.std(axis=1).replace(0, np.nan)
        cs_mean = raw.mean(axis=1)
        feat[f"mom_zscore_{n}d"] = raw.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # ── 15. Triple EMA crossover signals (3 features) ────────────────────────
    ema8  = close.ewm(span=8,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    feat["tema_cross_8_21"]  = (ema8 / (ema21 + 1e-8)) - 1
    feat["tema_cross_21_55"] = (ema21 / (ema55 + 1e-8)) - 1
    feat["tema_alignment"]   = (
        (ema8 > ema21).astype(float) + (ema21 > ema55).astype(float)
    ) / 2

    # ── 16. Momentum percentile rank within own history (2 features) ────────
    mom_20 = close.pct_change(20)
    mom_60 = close.pct_change(60)
    feat["mom_20d_ts_rank"] = mom_20.rank(axis=0, pct=True)
    feat["mom_60d_ts_rank"] = mom_60.rank(axis=0, pct=True)

    # ── 17. Cross-sectional momentum spread (long-short signal strength) (2) ─
    feat["mom_spread_12m"] = close.pct_change(252).sub(close.pct_change(252).mean(axis=1), axis=0)
    feat["mom_spread_1m"]  = close.pct_change(21).sub(close.pct_change(21).mean(axis=1), axis=0)

    # ── Stack all and return ──────────────────────────────────────────────────
    result = pd.concat(
        {k: v.stack() for k, v in feat.items()},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
