"""
technical.py
Technical Indicator Features — AlphaLens Signal Generation Agent
~40 features: classic technical indicators used as alpha signals.

Features:
  - RSI (multiple windows)
  - Bollinger Bands (position, width)
  - Stochastic Oscillator (%K, %D)
  - ADX (Average Directional Index) proxy
  - Ichimoku-style signals
  - Williams %R
  - CCI (Commodity Channel Index)
  - Rate-of-change oscillators
  - Trend strength indicators
"""

import pandas as pd
import numpy as np


def compute_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features.

    Args:
        prices: MultiIndex (date, ticker) with adj_close, high, low, returns.

    Returns:
        DataFrame with technical feature columns.
    """
    close = prices["adj_close"].unstack("ticker")
    high  = prices["high"].unstack("ticker") if "high" in prices.columns else close
    low   = prices["low"].unstack("ticker") if "low" in prices.columns else close

    feat: dict = {}

    # ── 1. RSI (4 features) ──────────────────────────────────────────────────
    delta = close.diff()
    for n in [7, 14, 21, 28]:
        gain = delta.clip(lower=0).rolling(n, min_periods=n // 2).mean()
        loss = (-delta.clip(upper=0)).rolling(n, min_periods=n // 2).mean()
        rs = gain / (loss + 1e-8)
        feat[f"rsi_{n}"] = 100 - (100 / (1 + rs))

    # ── 2. Bollinger Bands (6 features) ──────────────────────────────────────
    for n in [10, 20, 50]:
        sma = close.rolling(n, min_periods=n // 2).mean()
        std = close.rolling(n, min_periods=n // 2).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        feat[f"bb_pos_{n}"]   = (close - lower) / (upper - lower + 1e-8)
        feat[f"bb_width_{n}"] = (upper - lower) / (sma + 1e-8)

    # ── 3. Stochastic Oscillator (4 features) ────────────────────────────────
    for n in [14, 21]:
        roll_low  = low.rolling(n, min_periods=n // 2).min()
        roll_high = high.rolling(n, min_periods=n // 2).max()
        k = 100 * (close - roll_low) / (roll_high - roll_low + 1e-8)
        d = k.rolling(3, min_periods=2).mean()
        feat[f"stoch_k_{n}"] = k
        feat[f"stoch_d_{n}"] = d

    # ── 4. Williams %R (2 features) ──────────────────────────────────────────
    for n in [14, 28]:
        roll_low  = low.rolling(n, min_periods=n // 2).min()
        roll_high = high.rolling(n, min_periods=n // 2).max()
        feat[f"williams_r_{n}"] = -100 * (roll_high - close) / (roll_high - roll_low + 1e-8)

    # ── 5. CCI — Commodity Channel Index (2 features) ────────────────────────
    typical_price = (high + low + close) / 3
    for n in [14, 20]:
        sma_tp = typical_price.rolling(n, min_periods=n // 2).mean()
        mad = typical_price.rolling(n, min_periods=n // 2).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        feat[f"cci_{n}"] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)

    # ── 6. ADX proxy — trend strength (3 features) ───────────────────────────
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr        = (high - low).combine((high - close.shift(1)).abs(), np.maximum) \
                            .combine((low - close.shift(1)).abs(), np.maximum)
    for n in [14, 21]:
        atr      = tr.rolling(n, min_periods=n // 2).mean()
        plus_di  = 100 * plus_dm.rolling(n, min_periods=n // 2).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(n, min_periods=n // 2).mean() / (atr + 1e-8)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        feat[f"adx_{n}"] = dx.rolling(n, min_periods=n // 2).mean()
    feat["di_diff_14"] = plus_di - minus_di

    # ── 7. Ichimoku-style signals (4 features) ───────────────────────────────
    tenkan_high = high.rolling(9, min_periods=5).max()
    tenkan_low  = low.rolling(9, min_periods=5).min()
    tenkan_sen  = (tenkan_high + tenkan_low) / 2

    kijun_high = high.rolling(26, min_periods=13).max()
    kijun_low  = low.rolling(26, min_periods=13).min()
    kijun_sen  = (kijun_high + kijun_low) / 2

    feat["ichimoku_conv_base_diff"] = (tenkan_sen - kijun_sen) / (close + 1e-8)
    feat["ichimoku_price_vs_base"]  = (close - kijun_sen) / (close + 1e-8)

    senkou_a = (tenkan_sen + kijun_sen) / 2
    senkou_high = high.rolling(52, min_periods=26).max()
    senkou_low  = low.rolling(52, min_periods=26).min()
    senkou_b = (senkou_high + senkou_low) / 2
    feat["ichimoku_cloud_thickness"] = (senkou_a - senkou_b).abs() / (close + 1e-8)
    feat["ichimoku_price_above_cloud"] = (close > pd.concat([senkou_a, senkou_b]).groupby(level=0).max()).astype(float) \
        if False else ((close > senkou_a) & (close > senkou_b)).astype(float)

    # ── 8. Rate-of-change oscillators (4 features) ───────────────────────────
    for n in [9, 14, 25, 50]:
        feat[f"roc_osc_{n}"] = 100 * (close - close.shift(n)) / (close.shift(n) + 1e-8)

    # ── 9. Trend strength: linear regression slope (3 features) ──────────────
    def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        def slope(y):
            if np.isnan(y).any():
                return np.nan
            return ((x - x_mean) * (y - y.mean())).sum() / x_var
        return s.rolling(window, min_periods=window).apply(slope, raw=True)

    for n in [20, 60, 120]:
        feat[f"trend_slope_{n}"] = close.apply(lambda col: _rolling_slope(col, n)) / (close + 1e-8)

    # ── 10. Momentum oscillator divergence (2 features) ──────────────────────
    rsi_14 = feat["rsi_14"]
    feat["rsi_price_divergence"] = (
        close.pct_change(14).rank(axis=1, pct=True) - rsi_14.rank(axis=1, pct=True) / 100
    )
    feat["rsi_overbought"] = (rsi_14 > 70).astype(float)

    # ── 11. RSI oversold + extremes (2 features) ──────────────────────────────
    feat["rsi_oversold"]  = (rsi_14 < 30).astype(float)
    feat["rsi_zscore"]    = (rsi_14 - 50) / 25.0

    # ── 12. Bollinger Band squeeze and breakout (2 features) ─────────────────
    feat["bb_squeeze_20"]   = (
        feat["bb_width_20"] < feat["bb_width_20"].rolling(60, min_periods=30).quantile(0.2)
    ).astype(float)
    feat["bb_breakout_20"]  = (
        (feat["bb_pos_20"] > 1.0) | (feat["bb_pos_20"] < 0.0)
    ).astype(float)

    # ── 13. Chaikin-style money flow proxy (2 features) ───────────────────────
    if "high" in prices.columns and "low" in prices.columns:
        clv = ((close - low) - (high - close)) / ((high - low) + 1e-8)
        feat["money_flow_20"] = clv.rolling(20, min_periods=10).mean()
        feat["money_flow_60"] = clv.rolling(60, min_periods=30).mean()
    else:
        feat["money_flow_20"] = close.pct_change().rolling(20, min_periods=10).mean()
        feat["money_flow_60"] = close.pct_change().rolling(60, min_periods=30).mean()

    # ── 14. Aroon-style trend timing indicator (2 features) ───────────────────
    def _aroon_up(window: pd.Series, n: int) -> float:
        # Periods since the highest high in the window (0 = today is the high)
        idx_of_max = np.argmax(window.values)
        periods_since = (len(window) - 1) - idx_of_max
        return 100.0 * (n - periods_since) / n

    for n in [14, 25]:
        feat[f"aroon_up_{n}"] = high.rolling(n, min_periods=n).apply(
            lambda w: _aroon_up(w, n), raw=False
        )

    # ── Stack and return ──────────────────────────────────────────────────────
    result = pd.concat(
        {k: v.stack() for k, v in feat.items()},
        axis=1,
    )
    result.index.names = ["date", "ticker"]
    return result
