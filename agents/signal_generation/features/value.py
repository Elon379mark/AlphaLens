import pandas as pd
import numpy as np


def compute_value_features(prices: pd.DataFrame, fundamentals_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Compute value-based features.
    Input:
      prices: DataFrame indexed by (date, ticker), must have 'adj_close'.
      fundamentals_aligned: DataFrame indexed by (date, ticker), already
        forward-filled onto the price calendar via align_fundamentals_to_prices.
        Must NOT be raw quarterly fundamentals — misaligned dates will silently
        produce wrong ratios (this was a real bug in an earlier build).
    Returns: DataFrame indexed by (date, ticker) with 50 value feature columns.
    """
    # Defensive check: catch the exact bug class from project history early
    common_idx = prices.index.intersection(fundamentals_aligned.index)
    overlap_frac = len(common_idx) / len(prices.index) if len(prices.index) > 0 else 0
    if overlap_frac < 0.90:
        raise ValueError(
            f"Only {overlap_frac:.1%} of price index rows have matching fundamentals rows. "
            f"This usually means fundamentals were not aligned to the price calendar first "
            f"(missing call to align_fundamentals_to_prices, or wrong date union)."
        )

    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"]

    pe = fundamentals_aligned["pe_ratio"].reindex(prices.index)
    pb = fundamentals_aligned["pb_ratio"].reindex(prices.index)
    ev_ebitda = fundamentals_aligned["ev_ebitda"].reindex(prices.index)
    earnings_yield = fundamentals_aligned["earnings_yield"].reindex(prices.index)
    dividend_yield = fundamentals_aligned["dividend_yield"].reindex(prices.index)

    # --- 1. Raw ratios and their inverses (10 features) ---
    feats["pe_ratio"] = pe
    feats["earnings_yield_raw"] = earnings_yield
    feats["pb_ratio"] = pb
    feats["book_to_price"] = 1.0 / pb.replace(0, np.nan)
    feats["ev_ebitda"] = ev_ebitda
    feats["ebitda_to_ev"] = 1.0 / ev_ebitda.replace(0, np.nan)
    feats["dividend_yield"] = dividend_yield
    feats["pe_inverse"] = 1.0 / pe.replace(0, np.nan)
    feats["earnings_yield_x_dividend"] = earnings_yield * dividend_yield
    feats["value_composite_raw"] = (1.0 / pe.replace(0, np.nan)) + (1.0 / pb.replace(0, np.nan))

    # --- 2. Cross-sectional z-scores of each ratio (10 features) ---
    close_wide_idx = prices.index.get_level_values("date")
    for col, name in [
        (pe, "pe"), (pb, "pb"), (ev_ebitda, "ev_ebitda"),
        (earnings_yield, "earnings_yield"), (dividend_yield, "dividend_yield"),
    ]:
        grouped = col.groupby(level="date")
        z = (col - grouped.transform("mean")) / grouped.transform("std")
        feats[f"{name}_zscore"] = z
        feats[f"{name}_zscore_inv"] = -z  # cheap = high inverse z-score

    # --- 3. Cross-sectional percentile rank (10 features) ---
    for col, name in [
        (pe, "pe"), (pb, "pb"), (ev_ebitda, "ev_ebitda"),
        (earnings_yield, "earnings_yield"), (dividend_yield, "dividend_yield"),
    ]:
        rank_asc = col.groupby(level="date").rank(pct=True)
        feats[f"{name}_pct_rank"] = rank_asc
        feats[f"{name}_pct_rank_inv"] = 1.0 - rank_asc

    # --- 4. Value momentum: change in valuation over time (10 features) ---
    pe_wide = pe.unstack("ticker")
    pb_wide = pb.unstack("ticker")
    ev_wide = ev_ebitda.unstack("ticker")
    ey_wide = earnings_yield.unstack("ticker")
    dy_wide = dividend_yield.unstack("ticker")

    for n in [60, 120]:
        feats[f"pe_chg_{n}d"] = pe_wide.pct_change(n).stack()
        feats[f"pb_chg_{n}d"] = pb_wide.pct_change(n).stack()
        feats[f"ev_ebitda_chg_{n}d"] = ev_wide.pct_change(n).stack()
        feats[f"earnings_yield_chg_{n}d"] = ey_wide.diff(n).stack()
        feats[f"dividend_yield_chg_{n}d"] = dy_wide.diff(n).stack()

    # --- 5. Price relative to valuation-implied fair value proxies (10 features) ---
    for n in [20, 60, 120, 252, 90]:
        price_wide = close.unstack("ticker")
        price_ma = price_wide.rolling(n).mean()
        # "cheapness vs own history" combined with earnings yield level
        feats[f"price_vs_ma_x_ey_{n}d"] = (
            ((price_wide - price_ma) / price_ma) * (-1) * ey_wide
        ).stack()
        
        feats[f"pb_relative_to_history_{n}d"] = (
                pb_wide / pb_wide.rolling(n).mean()
        ).stack()

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import (
        load_ohlcv, load_fundamentals, align_fundamentals_to_prices,
    )

    print("Loading sample OHLCV data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")

    print("Loading sample fundamentals data...")
    fundamentals = load_fundamentals("data/processed/sample_fundamentals.parquet")

    print("Aligning fundamentals to price calendar...")
    price_dates = prices.index.get_level_values("date").unique().sort_values()
    tickers = prices.index.get_level_values("ticker").unique().tolist()
    fundamentals_aligned = align_fundamentals_to_prices(fundamentals, price_dates, tickers)
    print(f"Aligned fundamentals shape: {fundamentals_aligned.shape}")

    print("\nComputing value features...")
    feats = compute_value_features(prices, fundamentals_aligned)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of value features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 50, f"Expected 50 value features, got {feats.shape[1]}"
    print("\nPASS: exactly 50 value features generated")