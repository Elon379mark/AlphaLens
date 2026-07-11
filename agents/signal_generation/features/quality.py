import pandas as pd
import numpy as np


def compute_quality_features(prices: pd.DataFrame, fundamentals_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality-based features.
    Input:
      prices: DataFrame indexed by (date, ticker), must have 'adj_close'.
      fundamentals_aligned: DataFrame indexed by (date, ticker), already
        forward-filled onto the price calendar via align_fundamentals_to_prices.
    Returns: DataFrame indexed by (date, ticker) with 45 quality feature columns.
    """
    common_idx = prices.index.intersection(fundamentals_aligned.index)
    overlap_frac = len(common_idx) / len(prices.index) if len(prices.index) > 0 else 0
    if overlap_frac < 0.90:
        raise ValueError(
            f"Only {overlap_frac:.1%} of price index rows have matching fundamentals rows. "
            f"Fundamentals were likely not aligned via align_fundamentals_to_prices first."
        )

    feats = pd.DataFrame(index=prices.index)

    roe = fundamentals_aligned["roe"].reindex(prices.index)
    roa = fundamentals_aligned["roa"].reindex(prices.index)
    gross_margin = fundamentals_aligned["gross_margin"].reindex(prices.index)
    debt_equity = fundamentals_aligned["debt_equity"].reindex(prices.index)
    current_ratio = fundamentals_aligned["current_ratio"].reindex(prices.index)
    asset_growth = fundamentals_aligned["asset_growth"].reindex(prices.index)
    accruals = fundamentals_aligned["accruals"].reindex(prices.index)

    # --- 1. Raw profitability ratios (7 features) ---
    feats["roe"] = roe
    feats["roa"] = roa
    feats["gross_margin"] = gross_margin
    feats["debt_equity"] = debt_equity
    feats["current_ratio"] = current_ratio
    feats["asset_growth"] = asset_growth
    feats["accruals"] = accruals

    # --- 2. Quality composite scores (5 features) ---
    feats["profitability_composite"] = (roe.rank(pct=True) + roa.rank(pct=True) + gross_margin.rank(pct=True)) / 3
    feats["low_leverage_score"] = 1.0 / debt_equity.replace(0, np.nan)
    feats["liquidity_score"] = current_ratio
    feats["low_accruals_score"] = -accruals  # lower accruals = higher earnings quality
    feats["conservative_growth_score"] = -asset_growth.abs()  # penalize extreme growth (either direction)

    # --- 3. Cross-sectional z-scores (10 features, inverted where "lower is better") ---
    for col, name, invert in [
        (roe, "roe", False), (roa, "roa", False), (gross_margin, "gross_margin", False),
        (debt_equity, "debt_equity", True), (current_ratio, "current_ratio", False),
    ]:
        grouped = col.groupby(level="date")
        z = (col - grouped.transform("mean")) / grouped.transform("std")
        if invert:
            z = -z
        feats[f"{name}_zscore"] = z
        feats[f"{name}_zscore_rank"] = z.groupby(level="date").rank(pct=True)

    # --- 4. Cross-sectional percentile rank (7 features) ---
    for col, name in [
        (roe, "roe"), (roa, "roa"), (gross_margin, "gross_margin"),
        (current_ratio, "current_ratio"), (accruals, "accruals"),
        (asset_growth, "asset_growth"), (debt_equity, "debt_equity"),
    ]:
        feats[f"{name}_pct_rank"] = col.groupby(level="date").rank(pct=True)

    # --- 5. Quality trend: change in fundamentals over time (10 features) ---
    roe_wide = roe.unstack("ticker")
    roa_wide = roa.unstack("ticker")
    gm_wide = gross_margin.unstack("ticker")
    de_wide = debt_equity.unstack("ticker")
    cr_wide = current_ratio.unstack("ticker")

    for n in [60, 120]:
        feats[f"roe_chg_{n}d"] = roe_wide.diff(n).stack()
        feats[f"roa_chg_{n}d"] = roa_wide.diff(n).stack()
        feats[f"gross_margin_chg_{n}d"] = gm_wide.diff(n).stack()
        feats[f"debt_equity_chg_{n}d"] = de_wide.diff(n).stack()
        feats[f"current_ratio_chg_{n}d"] = cr_wide.diff(n).stack()

    # --- 6. Quality stability (rolling volatility of fundamentals, i.e. earnings consistency) (6 features) ---
    for n in [60, 120, 252]:
        feats[f"roe_stability_{n}d"] = -roe_wide.rolling(n).std().stack()  # lower vol = more stable = better
        feats[f"roa_stability_{n}d"] = -roa_wide.rolling(n).std().stack()

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

    print("\nComputing quality features...")
    feats = compute_quality_features(prices, fundamentals_aligned)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of quality features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 45, f"Expected 45 quality features, got {feats.shape[1]}"
    print("\nPASS: exactly 45 quality features generated")