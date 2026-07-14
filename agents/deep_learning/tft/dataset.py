import json
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

MAX_ENCODER_LENGTH = 20    # lookback window (kept short — our sample data is only 500 days)
MAX_PREDICTION_LENGTH = 21  # ~1 month forecast horizon, matches Chapter 3's forward return horizon
TOP_N_SIGNALS = 20          # number of top-ranked features from Chapter 3 to feed into TFT


def assign_synthetic_sector(ticker: str) -> str:
    """
    Our synthetic Chapter 3 dataset has no sector field. Deterministically
    assign a synthetic sector per ticker (stable across runs) so TFT has a
    static categorical to work with, matching the manual's architecture.
    Real deployment would replace this with actual GICS sector data.
    """
    sectors = ["Tech", "Healthcare", "Financials", "Energy", "Consumer", "Industrials"]
    hash_val = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    return sectors[hash_val % len(sectors)]


def load_top_signals(ranked_signals_path: str = "outputs/ranked_signals.json", top_n: int = TOP_N_SIGNALS) -> List[str]:
    """
    Load the top-N ranked signal names from Chapter 3's output, excluding
    features with very long lookback windows (>=120 days) in their name.
    Long-window features cost ~120-252 days of NaN warm-up per ticker via
    dropna(), which on our small 500-day sample destroys almost all usable
    training rows. This filter keeps the dataset usable; it can be relaxed
    once real, longer-history market data replaces the synthetic sample.
    """
    with open(ranked_signals_path) as f:
        ranked = json.load(f)

    long_window_markers = ["120d", "120", "252d", "252", "180d", "180", "150", "126", "189"]
    filtered = [
        s for s in ranked
        if not any(marker in s for marker in long_window_markers)
    ]
    return filtered[:top_n]


def prepare_tft_dataframe(
    features_path: str = "data/processed/features.parquet",
    ranked_signals_path: str = "outputs/ranked_signals.json",
    top_n: int = TOP_N_SIGNALS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load features, select top-N signals, and reshape into the flat format
    TimeSeriesDataSet expects: one row per (group_id, time_idx), with
    time_idx as a sequential integer per group (not a raw date).
    Returns (dataframe, list of feature column names used).
    """
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])

    top_signals = load_top_signals(ranked_signals_path, top_n)
    # Guard against any ranked signal names that might not exist as columns
    # (shouldn't happen if Chapter 3 ran correctly, but fail loudly if it did)
    missing = [s for s in top_signals if s not in features.columns]
    if missing:
        raise ValueError(f"Ranked signals not found in features.parquet: {missing}")

    keep_cols = ["date", "ticker"] + top_signals
    df = features[keep_cols].copy()

    # Build target: forward return, computed the same way as Chapter 3's IC calculator
    df = df.sort_values(["ticker", "date"])
    df["close_placeholder"] = 1.0  # not used for target directly; target computed below via features already reflecting price info is NOT valid -- compute properly:

    # We need actual forward returns as the TFT target. Recompute from raw prices
    # directly rather than trying to reverse-engineer it from features.
    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["fwd_return_21d"] = (
        prices.groupby("ticker")["adj_close"].shift(-MAX_PREDICTION_LENGTH)
        / prices["adj_close"] - 1
    )

    df = df.merge(prices[["date", "ticker", "fwd_return_21d"]], on=["date", "ticker"], how="left")
    df = df.drop(columns=["close_placeholder"])

    # Static categoricals
    df["sector"] = df["ticker"].apply(assign_synthetic_sector)

    # Time-varying known reals — encoded numerically (not as string categories)
    # to avoid "unknown category" errors when a validation/prediction window
    # contains a calendar month or weekday that never appeared in the smaller
    # training window (a real risk with our short synthetic sample period).
    df["month"] = df["date"].dt.month.astype(float)
    df["weekday"] = df["date"].dt.weekday.astype(float)

   # Drop rows with no target (last MAX_PREDICTION_LENGTH days per ticker have no forward return)
    df = df.dropna(subset=["fwd_return_21d"])

    # Drop rows with NaN in any feature column (TFT requires clean numeric input;
    # this reflects the same warm-up NaN issue seen throughout Chapter 3 features)
    before = len(df)
    df = df.dropna(subset=top_signals)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN features (feature warm-up period)")

    # IMPORTANT: recompute time_idx AFTER dropping NaN rows, not before.
    # Assigning it earlier leaves gaps wherever a scattered NaN row was removed
    # from the middle of a ticker's series (not just the leading warm-up block).
    # TimeSeriesDataSet's allow_missing_timesteps=True then auto-fills those gaps
    # with placeholder rows that have no real target value, causing NaN target
    # errors downstream. Recomputing here guarantees a contiguous 0..N sequence
    # per ticker across only the rows we're actually keeping.
    df = df.sort_values(["ticker", "date"])
    df["time_idx"] = df.groupby("ticker").cumcount()

    return df, top_signals


def build_tft_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str = "fwd_return_21d") -> TimeSeriesDataSet:
    """
    Build a PyTorch Forecasting TimeSeriesDataSet from the prepared dataframe.
    """
    max_prediction_length = MAX_PREDICTION_LENGTH
    max_encoder_length = MAX_ENCODER_LENGTH

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,

        static_categoricals=["sector", "ticker"],

        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx", "month", "weekday"],

        time_varying_unknown_reals=feature_cols + [target_col],

        target_normalizer=GroupNormalizer(groups=["ticker"]),  # no transformation — target is a return, can be negative; softplus assumes non-negative targets and produces NaN/inf here
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    return dataset


if __name__ == "__main__":
    print("Preparing TFT dataframe from Chapter 3 outputs...")
    df, feature_cols = prepare_tft_dataframe()

    print(f"\nPrepared dataframe shape: {df.shape}")
    print(f"Feature columns used ({len(feature_cols)}): {feature_cols}")
    print(f"\nTickers: {df['ticker'].nunique()}")
    print(f"Time range: {df['date'].min()} to {df['date'].max()}")
    print(f"time_idx range per ticker: 0 to {df['time_idx'].max()}")

    print(f"\nSample rows:")
    print(df[["date", "ticker", "time_idx", "sector", "fwd_return_21d"] + feature_cols[:3]].head())

    print("\nBuilding TimeSeriesDataSet...")
    tft_dataset = build_tft_dataset(df, feature_cols)

    print(f"\nDataset built successfully.")
    print(f"Number of samples: {len(tft_dataset)}")

    assert len(tft_dataset) > 0, "TimeSeriesDataSet has zero samples — check encoder/prediction length vs. available data"
    print("\nPASS: TFT dataset prepared and built successfully")