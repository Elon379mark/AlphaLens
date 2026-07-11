import pandas as pd
import numpy as np


def compute_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-based features from OHLCV data.
    Input: prices DataFrame indexed by (date, ticker), needs adj_close, high, low.
    Returns: DataFrame indexed by (date, ticker) with 40 volatility feature columns.
    """
    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"].unstack("ticker")
    high = prices["high"].unstack("ticker")
    low = prices["low"].unstack("ticker")
    daily_ret = close.pct_change()

    # --- 1. Realized volatility over multiple windows (8 features) ---
    for n in [5, 10, 20, 40, 60, 90, 120, 252]:
        feats[f"realized_vol_{n}d"] = (daily_ret.rolling(n).std() * np.sqrt(252)).stack()

    # --- 2. Vol-of-vol: volatility of realized volatility (5 features) ---
    for n in [20, 40, 60, 90, 120]:
        rolling_vol = daily_ret.rolling(20).std()
        feats[f"vol_of_vol_{n}d"] = rolling_vol.rolling(n).std().stack()

    # --- 3. Downside deviation (semi-deviation, only negative returns) (5 features) ---
    for n in [20, 40, 60, 90, 120]:
        downside_ret = daily_ret.where(daily_ret < 0, 0)
        feats[f"downside_dev_{n}d"] = (downside_ret.rolling(n).std() * np.sqrt(252)).stack()

    # --- 4. Average True Range (ATR) family (5 features) ---
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).stack(),
        (high.stack() - prev_close.stack()).abs(),
        (low.stack() - prev_close.stack()).abs(),
    ], axis=1).max(axis=1)
    tr_wide = tr.unstack("ticker")
    for n in [5, 10, 20, 40, 60]:
        feats[f"atr_{n}d"] = tr_wide.rolling(n).mean().stack()

    # --- 5. Normalized ATR (ATR as % of price) (5 features) ---
    for n in [5, 10, 20, 40, 60]:
        atr_n = tr_wide.rolling(n).mean()
        feats[f"atr_pct_{n}d"] = (atr_n / close).stack()

    # --- 6. Rolling beta vs cross-sectional market proxy (5 features) ---
    market_ret = daily_ret.mean(axis=1)  # equal-weighted proxy for "market"
    for n in [60, 90, 120, 252, 180]:
        cov = daily_ret.rolling(n).cov(market_ret)
        market_var = market_ret.rolling(n).var()
        feats[f"beta_{n}d"] = cov.div(market_var, axis=0).stack()

    # --- 7. High-low range volatility (5 features) ---
    hl_range = ((high - low) / close)
    for n in [5, 10, 20, 40, 60]:
        feats[f"hl_range_vol_{n}d"] = hl_range.rolling(n).mean().stack()

    # --- 8. Volatility regime: current vol vs long-term average vol (2 features) ---
    vol_20 = daily_ret.rolling(20).std()
    vol_120 = daily_ret.rolling(120).std()
    vol_252 = daily_ret.rolling(252).std()
    feats["vol_regime_short_vs_long"] = (vol_20 / vol_120).stack()
    feats["vol_regime_short_vs_year"] = (vol_20 / vol_252).stack()

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import load_ohlcv

    print("Loading sample OHLCV data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")

    print("\nComputing volatility features...")
    feats = compute_volatility_features(prices)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of volatility features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 40, f"Expected 40 volatility features, got {feats.shape[1]}"
    print("\nPASS: exactly 40 volatility features generated")