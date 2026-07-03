"""
data_loader.py
Data Loading — AlphaLens Signal Generation Agent
Loads and aligns OHLCV and fundamental data into MultiIndex DataFrames.
Index convention: (date, ticker) throughout the signal generation pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a Parquet file.

    Expected columns in file: date, ticker, open, high, low, close, volume, adj_close.
    Returns DataFrame with MultiIndex (date, ticker).
    Columns: open, high, low, close, volume, adj_close, returns.
    """
    df = pd.read_parquet(path)

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    df = df.set_index(["date", "ticker"]).sort_index()

    # Compute daily returns per ticker
    adj = df["adj_close"].unstack("ticker")
    returns = adj.pct_change()
    df["returns"] = returns.stack()

    return df


def load_fundamentals(path: str) -> pd.DataFrame:
    """
    Load fundamental data from a Parquet file.

    Expected columns: ticker, date, pe_ratio, pb_ratio, roe, roa, gross_margin,
    earnings_yield, debt_equity, current_ratio, asset_growth, accruals,
    ev_ebitda, dividend_yield.
    Returns DataFrame indexed by (date, ticker).
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def align_data(
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join OHLCV with fundamentals on (date, ticker) index.
    Fundamental data is forward-filled within each ticker group
    to handle quarterly reporting frequency.

    Forward-fill is performed on the union of fundamentals report dates and
    OHLCV trading dates, then sliced back down to OHLCV dates. Reindexing
    directly to OHLCV dates before ffill would drop every fundamentals
    observation (report dates rarely coincide with trading dates), leaving
    nothing to forward-fill from.

    Returns merged DataFrame.
    """
    ohlcv_dates = ohlcv.unstack("ticker").index
    fund_wide = fundamentals.unstack("ticker")
    union_dates = fund_wide.index.union(ohlcv_dates).sort_values()

    fund_ffill = (
        fund_wide
        .reindex(union_dates)
        .ffill()
        .reindex(ohlcv_dates)
        .stack("ticker")
        .reindex(ohlcv.index)
    )
    merged = ohlcv.join(fund_ffill, how="left")
    return merged


def create_sample_data(
    n_tickers: int = 50,
    n_days: int = 1260,  # ~5 years
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic OHLCV + fundamentals for testing.
    Returns (ohlcv_df, fundamentals_df).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.bdate_range(end="2024-01-31", periods=n_days)

    records = []
    for ticker in tickers:
        # Random walk for prices
        log_returns = rng.normal(0.0003, 0.015, n_days)
        price = 100.0 * np.exp(np.cumsum(log_returns))
        volume = rng.integers(100_000, 10_000_000, n_days).astype(float)
        for i, dt in enumerate(dates):
            p = price[i]
            noise = rng.uniform(0.98, 1.02)
            records.append({
                "date": dt,
                "ticker": ticker,
                "open": p * rng.uniform(0.99, 1.01),
                "high": p * rng.uniform(1.00, 1.02),
                "low": p * rng.uniform(0.98, 1.00),
                "close": p * noise,
                "adj_close": p * noise,
                "volume": volume[i],
            })

    ohlcv = pd.DataFrame(records)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv.set_index(["date", "ticker"]).sort_index()
    adj = ohlcv["adj_close"].unstack("ticker")
    ohlcv["returns"] = adj.pct_change().stack()

    # Fundamentals (quarterly cadence, forward-filled)
    fund_records = []
    n_quarters = max(8, (n_days // 63) + 4)  # enough quarters to cover the OHLCV range + buffer
    quarter_dates = pd.date_range(end="2024-01-31", periods=n_quarters, freq="QS")
    for ticker in tickers:
        for dt in quarter_dates:
            fund_records.append({
                "date": dt,
                "ticker": ticker,
                "pe_ratio": rng.uniform(5, 40),
                "pb_ratio": rng.uniform(0.5, 8),
                "roe": rng.uniform(-0.05, 0.35),
                "roa": rng.uniform(-0.02, 0.15),
                "gross_margin": rng.uniform(0.1, 0.7),
                "earnings_yield": rng.uniform(0.01, 0.12),
                "debt_equity": rng.uniform(0.1, 3.0),
                "current_ratio": rng.uniform(0.5, 4.0),
                "asset_growth": rng.uniform(-0.1, 0.3),
                "accruals": rng.uniform(-0.1, 0.1),
                "ev_ebitda": rng.uniform(3, 25),
                "dividend_yield": rng.uniform(0, 0.06),
                "log_market_cap": rng.uniform(20, 27),
                "sector": rng.choice(["Tech", "Finance", "Health", "Energy", "Consumer"]),
            })

    fundamentals = pd.DataFrame(fund_records)
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])
    fundamentals = fundamentals.set_index(["date", "ticker"]).sort_index()

    return ohlcv, fundamentals
