import pandas as pd
import numpy as np


def compute_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum-based features from OHLCV data.
    Input: prices DataFrame indexed by (date, ticker) with an 'adj_close' column.
    Returns: DataFrame indexed by (date, ticker) with 60 momentum feature columns.
    """
    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"].unstack("ticker")

    # --- 1. Simple N-day returns (10 features) ---
    lookbacks = [5, 10, 15, 20, 30, 40, 60, 90, 120, 252]
    for n in lookbacks:
        feats[f"mom_{n}d"] = close.pct_change(n).stack()

    # --- 2. Skip-month momentum variants (6 features) ---
    # Classic 12-1, 6-1, plus additional skip combinations
    feats["mom_12_1"] = (close.pct_change(252) - close.pct_change(21)).stack()
    feats["mom_6_1"] = (close.pct_change(126) - close.pct_change(21)).stack()
    feats["mom_9_1"] = (close.pct_change(189) - close.pct_change(21)).stack()
    feats["mom_3_1"] = (close.pct_change(63) - close.pct_change(21)).stack()
    feats["mom_12_2"] = (close.pct_change(252) - close.pct_change(42)).stack()
    feats["mom_6_2"] = (close.pct_change(126) - close.pct_change(42)).stack()

    # --- 3. MACD family (8 features) ---
    macd_configs = [(12, 26, 9), (5, 35, 5), (8, 17, 9), (19, 39, 9)]
    for fast, slow, sig in macd_configs:
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=sig).mean()
        feats[f"macd_hist_{fast}_{slow}_{sig}"] = (macd - signal_line).stack()
        feats[f"macd_cross_{fast}_{slow}_{sig}"] = (
            (macd > signal_line).astype(int) - (macd.shift(1) > signal_line.shift(1)).astype(int)
        ).stack()

    # --- 4. Momentum acceleration / deceleration (6 features) ---
    accel_pairs = [(10, 20), (20, 40), (40, 80), (60, 120), (5, 10), (15, 30)]
    for short, long in accel_pairs:
        feats[f"mom_accel_{short}_{long}"] = (
            close.pct_change(short) - close.pct_change(long)
        ).stack()

    # --- 5. Residual / relative momentum vs cross-sectional mean (6 features) ---
    for n in [20, 60, 120, 252, 40, 90]:
        period_ret = close.pct_change(n)
        cross_mean = period_ret.mean(axis=1)
        feats[f"residual_mom_{n}d"] = period_ret.sub(cross_mean, axis=0).stack()

    # --- 6. Exponentially weighted momentum (6 features) ---
    for span in [10, 20, 40, 60, 90, 120]:
        ewm_ret = close.pct_change().ewm(span=span).mean()
        feats[f"ewm_mom_{span}"] = ewm_ret.stack()

    # --- 7. Momentum volatility-scaled (risk-adjusted momentum) (6 features) ---
    for n in [20, 60, 120, 252, 40, 90]:
        period_ret = close.pct_change(n)
        vol = close.pct_change().rolling(n).std()
        feats[f"risk_adj_mom_{n}d"] = (period_ret / (vol * np.sqrt(n))).stack()

    # --- 8. Distance from moving average (momentum proxy) (6 features) ---
    for n in [20, 50, 100, 200, 10, 150]:
        ma = close.rolling(n).mean()
        feats[f"dist_from_ma_{n}"] = ((close - ma) / ma).stack()

    # --- 9. New high / new low proximity (6 features) ---
    for n in [20, 60, 120, 252, 40, 90]:
        rolling_max = close.rolling(n).max()
        feats[f"pct_from_high_{n}d"] = ((close - rolling_max) / rolling_max).stack()

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import load_ohlcv

    print("Loading sample OHLCV data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")
    print(f"Input shape: {prices.shape}")

    print("\nComputing momentum features...")
    feats = compute_momentum_features(prices)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of momentum features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")
    print("(High NaN fraction near the start of the series is expected — long lookback windows need warm-up data)")

    assert feats.shape[1] == 60, f"Expected 60 momentum features, got {feats.shape[1]}"
    print("\nPASS: exactly 60 momentum features generated")