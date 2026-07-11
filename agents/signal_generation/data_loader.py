import numpy as np
import pandas as pd
from pathlib import Path


def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a parquet file.
    Expects columns: date, ticker, open, high, low, close, volume, adj_close.
    Returns DataFrame indexed by MultiIndex (date, ticker) with a 'returns' column added.
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()
    df["returns"] = df.groupby(level="ticker")["adj_close"].pct_change()
    return df


def load_fundamentals(path: str) -> pd.DataFrame:
    """
    Load fundamental data from a parquet file.
    Expects columns: ticker, date, pe_ratio, pb_ratio, roe, roa, gross_margin,
    earnings_yield, debt_equity, current_ratio, asset_growth, accruals,
    ev_ebitda, dividend_yield.
    Returns a flat DataFrame (NOT indexed) — callers are responsible for
    aligning dates against OHLCV before use, since fundamentals report at a
    lower frequency (quarterly) than prices (daily).
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def align_fundamentals_to_prices(
    fundamentals: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    tickers: list,
) -> pd.DataFrame:
    """
    Forward-fill fundamentals onto the daily price calendar.

    CRITICAL: fundamentals report quarterly, prices report daily. Naively
    reindexing fundamentals onto price dates without first UNIONING the two
    date sets causes silent misalignment (a real bug hit in an earlier build
    of this project) — forward-fill must happen on the union of both date
    sets, then be sliced back down to just the price dates.
    """
    result_frames = []
    for ticker in tickers:
        tick_fund = fundamentals[fundamentals["ticker"] == ticker].set_index("date").sort_index()
        if tick_fund.empty:
            continue

        # Union fundamentals dates + price dates so forward-fill has full history
        union_dates = tick_fund.index.union(price_dates).sort_values()
        tick_fund_reindexed = tick_fund.reindex(union_dates).ffill()

        # Slice back down to only the price calendar dates
        tick_fund_on_prices = tick_fund_reindexed.reindex(price_dates)
        tick_fund_on_prices["ticker"] = ticker
        result_frames.append(tick_fund_on_prices)

    if not result_frames:
        return pd.DataFrame()

    combined = pd.concat(result_frames)
    combined = combined.reset_index().rename(columns={"index": "date"})
    combined = combined.set_index(["date", "ticker"]).sort_index()
    return combined


def generate_sample_data(
    n_tickers: int = 20,
    n_days: int = 500,
    seed: int = 42,
    output_dir: str = "data/processed",
) -> None:
    """
    Generate synthetic OHLCV and fundamentals data for development/testing.
    Writes sample_prices.parquet and sample_fundamentals.parquet.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"TICK{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # --- OHLCV ---
    price_rows = []
    for ticker in tickers:
        start_price = rng.uniform(20, 500)
        prices = [start_price]
        for _ in range(n_days - 1):
            daily_ret = rng.normal(0.0003, 0.02)
            prices.append(prices[-1] * (1 + daily_ret))
        prices = np.array(prices)

        for i, date in enumerate(dates):
            close = prices[i]
            open_ = close * (1 + rng.normal(0, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.uniform(1e5, 5e6))
            price_rows.append({
                "date": date, "ticker": ticker,
                "open": open_, "high": high, "low": low,
                "close": close, "adj_close": close, "volume": volume,
            })

    prices_df = pd.DataFrame(price_rows)

    # --- Fundamentals (quarterly, not daily) ---
    quarter_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="QE")
    fund_rows = []
    for ticker in tickers:
        for date in quarter_dates:
            fund_rows.append({
                "date": date, "ticker": ticker,
                "pe_ratio": rng.uniform(8, 35),
                "pb_ratio": rng.uniform(0.5, 8),
                "roe": rng.uniform(-0.1, 0.35),
                "roa": rng.uniform(-0.05, 0.15),
                "gross_margin": rng.uniform(0.1, 0.6),
                "earnings_yield": rng.uniform(0.01, 0.12),
                "debt_equity": rng.uniform(0.1, 2.5),
                "current_ratio": rng.uniform(0.5, 3.0),
                "asset_growth": rng.uniform(-0.1, 0.3),
                "accruals": rng.uniform(-0.05, 0.05),
                "ev_ebitda": rng.uniform(4, 25),
                "dividend_yield": rng.uniform(0.0, 0.06),
            })

    fundamentals_df = pd.DataFrame(fund_rows)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(f"{output_dir}/sample_prices.parquet", index=False)
    fundamentals_df.to_parquet(f"{output_dir}/sample_fundamentals.parquet", index=False)

    print(f"Generated {len(prices_df)} price rows for {n_tickers} tickers over {n_days} days")
    print(f"Generated {len(fundamentals_df)} fundamental rows ({len(quarter_dates)} quarters)")
    print(f"Saved to {output_dir}/sample_prices.parquet and {output_dir}/sample_fundamentals.parquet")


if __name__ == "__main__":
    print("Generating synthetic sample data...")
    generate_sample_data()

    print("\nLoading OHLCV...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")
    print(f"OHLCV shape: {prices.shape}")
    print(prices.head())

    print("\nLoading fundamentals...")
    fundamentals = load_fundamentals("data/processed/sample_fundamentals.parquet")
    print(f"Fundamentals shape: {fundamentals.shape}")
    print(fundamentals.head())

    print("\nAligning fundamentals to price calendar...")
    price_dates = prices.index.get_level_values("date").unique().sort_values()
    tickers = prices.index.get_level_values("ticker").unique().tolist()
    aligned = align_fundamentals_to_prices(fundamentals, price_dates, tickers)
    print(f"Aligned fundamentals shape: {aligned.shape}")
    print(f"Expected shape: ({len(price_dates) * len(tickers)}, ...)")

    # Sanity check: no fully-empty rows after alignment (forward-fill should have covered nearly all)
    nan_frac = aligned["pe_ratio"].isna().mean()
    print(f"\nNaN fraction in pe_ratio after alignment: {nan_frac:.2%}")
    assert nan_frac < 0.15, "Too many NaNs after fundamentals alignment — check date union logic"
    print("PASS: fundamentals alignment looks correct")