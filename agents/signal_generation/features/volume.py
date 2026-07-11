import pandas as pd
import numpy as np


def compute_volume_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume-based features from OHLCV data.
    Input: prices DataFrame indexed by (date, ticker), needs adj_close, volume, high, low.
    Returns: DataFrame indexed by (date, ticker) with 35 volume feature columns.
    """
    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"].unstack("ticker")
    volume = prices["volume"].unstack("ticker")
    high = prices["high"].unstack("ticker")
    low = prices["low"].unstack("ticker")
    daily_ret = close.pct_change()

    # --- 1. Volume moving averages / trend (6 features) ---
    for n in [5, 10, 20, 40, 60, 90]:
        feats[f"volume_ma_{n}d"] = volume.rolling(n).mean().stack()

    # --- 2. Volume relative to its own history (5 features) ---
    for n in [20, 40, 60, 90, 120]:
        vol_ma = volume.rolling(n).mean()
        feats[f"volume_ratio_{n}d"] = (volume / vol_ma).stack()

    # --- 3. On-Balance Volume (OBV) and its trend (4 features) ---
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    feats["obv"] = obv.stack()
    for n in [20, 60, 120]:
        obv_ma = obv.rolling(n).mean()
        feats[f"obv_vs_ma_{n}d"] = ((obv - obv_ma) / obv_ma.abs().replace(0, np.nan)).stack()

    # --- 4. VWAP deviation (approximate, using typical price) (5 features) ---
    typical_price = (high + low + close) / 3
    for n in [5, 10, 20, 40, 60]:
        vwap = (typical_price * volume).rolling(n).sum() / volume.rolling(n).sum()
        feats[f"vwap_dev_{n}d"] = ((close - vwap) / vwap).stack()

    # --- 5. Amihud illiquidity (price impact per unit volume) (5 features) ---
    dollar_volume = close * volume
    illiq_daily = daily_ret.abs() / dollar_volume.replace(0, np.nan)
    for n in [20, 40, 60, 90, 120]:
        feats[f"amihud_illiq_{n}d"] = illiq_daily.rolling(n).mean().stack()

    # --- 6. Turnover (volume relative to a rolling average, proxy for shares outstanding) (5 features) ---
    for n in [20, 40, 60, 90, 120]:
        turnover_proxy = volume / volume.rolling(252).mean()
        feats[f"turnover_{n}d"] = turnover_proxy.rolling(n).mean().stack()

    # --- 7. Price-volume correlation (does volume confirm price direction?) (5 features) ---
    for n in [20, 40, 60, 90, 120]:
        feats[f"price_volume_corr_{n}d"] = daily_ret.rolling(n).corr(volume.pct_change()).stack()

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import load_ohlcv

    print("Loading sample OHLCV data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")

    print("\nComputing volume features...")
    feats = compute_volume_features(prices)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of volume features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 35, f"Expected 35 volume features, got {feats.shape[1]}"
    print("\nPASS: exactly 35 volume features generated")