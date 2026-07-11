import pandas as pd
import numpy as np


def compute_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute classical technical indicator features from OHLCV data.
    Input: prices DataFrame indexed by (date, ticker), needs adj_close, high, low.
    Returns: DataFrame indexed by (date, ticker) with 40 technical feature columns.
    """
    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"].unstack("ticker")
    high = prices["high"].unstack("ticker")
    low = prices["low"].unstack("ticker")

    # --- 1. RSI (Relative Strength Index) at multiple windows (6 features) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    for n in [7, 14, 21, 28, 42, 60]:
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        feats[f"rsi_{n}"] = rsi.stack()

    # --- 2. Bollinger Bands: %B and bandwidth (8 features) ---
    for n in [10, 20, 40, 60]:
        ma = close.rolling(n).mean()
        std = close.rolling(n).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
        bandwidth = (upper - lower) / ma.replace(0, np.nan)
        feats[f"bb_pct_b_{n}"] = pct_b.stack()
        feats[f"bb_bandwidth_{n}"] = bandwidth.stack()

    # --- 3. Stochastic Oscillator %K and %D (8 features) ---
    for n in [14, 21, 28, 42]:
        lowest_low = low.rolling(n).min()
        highest_high = high.rolling(n).max()
        pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        pct_d = pct_k.rolling(3).mean()
        feats[f"stoch_k_{n}"] = pct_k.stack()
        feats[f"stoch_d_{n}"] = pct_d.stack()

    # --- 4. ADX (Average Directional Index) proxy at multiple windows (4 features) ---
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    tr = pd.concat([
        (high - low).stack(),
        (high.stack() - close.shift(1).stack()).abs(),
        (low.stack() - close.shift(1).stack()).abs(),
    ], axis=1).max(axis=1).unstack("ticker")
    for n in [14, 21, 28, 42]:
        atr_n = tr.rolling(n).mean()
        plus_di = 100 * (plus_dm.rolling(n).mean() / atr_n.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(n).mean() / atr_n.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(n).mean()
        feats[f"adx_{n}"] = adx.stack()

    # --- 5. Ichimoku-style conversion/base line deviation (6 features) ---
    for n_conv, n_base in [(9, 26), (7, 22), (5, 18)]:
        conv_line = (high.rolling(n_conv).max() + low.rolling(n_conv).min()) / 2
        base_line = (high.rolling(n_base).max() + low.rolling(n_base).min()) / 2
        feats[f"ichimoku_conv_dev_{n_conv}_{n_base}"] = ((close - conv_line) / conv_line).stack()
        feats[f"ichimoku_base_dev_{n_conv}_{n_base}"] = ((close - base_line) / base_line).stack()

    # --- 6. Williams %R (4 features) ---
    for n in [14, 21, 28, 42]:
        highest_high = high.rolling(n).max()
        lowest_low = low.rolling(n).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
        feats[f"williams_r_{n}"] = williams_r.stack()

    # --- 7. Commodity Channel Index (CCI) (4 features) ---
    typical_price = (high + low + close) / 3
    for n in [14, 20, 28, 40]:
        tp_ma = typical_price.rolling(n).mean()
        mean_dev = typical_price.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (typical_price - tp_ma) / (0.015 * mean_dev.replace(0, np.nan))
        feats[f"cci_{n}"] = cci.stack()

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import load_ohlcv

    print("Loading sample OHLCV data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")

    print("\nComputing technical features (this one is heavier — CCI uses a rolling apply, may take a bit longer)...")
    feats = compute_technical_features(prices)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of technical features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 40, f"Expected 40 technical features, got {feats.shape[1]}"
    print("\nPASS: exactly 40 technical features generated")