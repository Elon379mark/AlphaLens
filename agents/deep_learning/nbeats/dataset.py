from typing import Tuple

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

MAX_ENCODER_LENGTH = 30    # N-BEATS lookback window
MAX_PREDICTION_LENGTH = 21  # matches Chapter 3's forward return horizon


def prepare_nbeats_dataframe(prices_path: str = "data/processed/sample_prices.parquet") -> pd.DataFrame:
    """
    N-BEATS is univariate — it forecasts a target series from its own history
    alone, with no auxiliary covariates. We build the return series directly
    from prices, without needing Chapter 3's feature matrix at all.
    """
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    prices["daily_return"] = prices.groupby("ticker")["adj_close"].pct_change()
    prices = prices.dropna(subset=["daily_return"])

    # Contiguous per-ticker time index (same lesson learned from TFT: compute
    # AFTER any row-dropping, not before, to avoid gaps)
    prices = prices.sort_values(["ticker", "date"])
    prices["time_idx"] = prices.groupby("ticker").cumcount()

    return prices[["date", "ticker", "time_idx", "daily_return"]].copy()


def build_nbeats_dataset(df: pd.DataFrame, target_col: str = "daily_return") -> TimeSeriesDataSet:
    """
    Build a TimeSeriesDataSet for N-BEATS: target-only, no covariates.
    """
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["ticker"],
        min_encoder_length=MAX_ENCODER_LENGTH,  # N-BEATS requires a FIXED encoder length (unlike TFT's variable-length support), so min must equal max
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,  # N-BEATS also requires a FIXED prediction length, same reasoning as encoder length above
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=[target_col],
        # No explicit target_normalizer here — GroupNormalizer silently adds
        # {target}_center / {target}_scale as extra static reals (confirmed via
        # diagnostic), which violates N-BEATS's strict "target-only" input
        # requirement. Leaving this as the 'auto' default (as the official
        # pytorch-forecasting N-BEATS tutorial does) avoids that.
        add_relative_time_idx=False,  # N-BEATS requires this False (unlike TFT)
        add_target_scales=False,      # N-BEATS also requires this False
        allow_missing_timesteps=True,
    )
    return dataset


if __name__ == "__main__":
    print("Preparing N-BEATS dataframe from raw prices (univariate — no Chapter 3 features needed)...")
    df = prepare_nbeats_dataframe()

    print(f"\nPrepared dataframe shape: {df.shape}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"time_idx range per ticker: 0 to {df['time_idx'].max()}")
    print(f"\nSample rows:")
    print(df.head())

    print("\nBuilding TimeSeriesDataSet...")
    dataset = build_nbeats_dataset(df)

    print(f"\nDataset built successfully.")
    print(f"Number of samples: {len(dataset)}")

    assert len(dataset) > 0, "N-BEATS dataset has zero samples"
    print("\nPASS: N-BEATS dataset prepared and built successfully")