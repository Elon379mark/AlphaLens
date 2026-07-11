import pandas as pd
import numpy as np


def compute_alternative_features(prices: pd.DataFrame, fundamentals_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Compute alternative-data-style features.

    IMPORTANT: These are PROXIES, not real alternative data. Real short interest,
    options IV, and analyst revisions require paid data feeds (Quandl, Refinitiv,
    etc. per the manual's Data Sources table) not yet integrated into this project.
    These proxies are derived from price/volume/fundamentals to fill the same
    functional role (crowding/sentiment/revision signals) until real feeds are added.

    Input:
      prices: DataFrame indexed by (date, ticker), needs adj_close, volume, high, low.
      fundamentals_aligned: DataFrame indexed by (date, ticker), pre-aligned to price calendar.
    Returns: DataFrame indexed by (date, ticker) with 30 alternative feature columns.
    """
    common_idx = prices.index.intersection(fundamentals_aligned.index)
    overlap_frac = len(common_idx) / len(prices.index) if len(prices.index) > 0 else 0
    if overlap_frac < 0.90:
        raise ValueError(
            f"Only {overlap_frac:.1%} of price index rows have matching fundamentals rows. "
            f"Fundamentals were likely not aligned via align_fundamentals_to_prices first."
        )

    feats = pd.DataFrame(index=prices.index)
    close = prices["adj_close"].unstack("ticker")
    volume = prices["volume"].unstack("ticker")
    high = prices["high"].unstack("ticker")
    low = prices["low"].unstack("ticker")
    daily_ret = close.pct_change()

    # --- 1. Short-interest-style crowding proxy: sustained volume spikes on down days (6 features) ---
    # Real short interest measures shares sold short; as a proxy, we look for
    # persistent high-volume selling pressure, which correlates with crowded/bearish positioning.
    down_day_volume = volume.where(daily_ret < 0, 0)
    for n in [10, 20, 40, 60, 90, 120]:
        avg_vol = volume.rolling(n).mean()
        crowding_proxy = (down_day_volume.rolling(n).sum() / avg_vol.rolling(n).sum().replace(0, np.nan))
        feats[f"short_crowding_proxy_{n}d"] = crowding_proxy.stack()

    # --- 2. Options-IV-style proxy: realized vol term structure (implied vol not available, use realized vol slope) (6 features) ---
    # Real options IV reflects market's forward-looking vol expectation. As a proxy,
    # we use the term structure slope of realized vol (short vs long window) — a
    # steepening slope often precedes IV term structure moves in real markets.
    vol_short = daily_ret.rolling(10).std()
    for n_long in [30, 60, 90, 120, 180, 252]:
        vol_long = daily_ret.rolling(n_long).std()
        feats[f"vol_term_structure_proxy_{n_long}d"] = (vol_short / vol_long.replace(0, np.nan)).stack()

    # --- 3. Analyst-revision-style proxy: fundamentals surprise/change momentum (6 features) ---
    # Real analyst revisions track changes in earnings estimates. As a proxy, we
    # use the rate of change in reported fundamentals (ROE, earnings yield) as a
    # stand-in for "positive/negative fundamental surprise momentum."
    roe = fundamentals_aligned["roe"].reindex(prices.index).unstack("ticker")
    ey = fundamentals_aligned["earnings_yield"].reindex(prices.index).unstack("ticker")
    for n in [40, 60, 90, 120, 180, 252]:
        roe_revision = roe.diff(n)
        feats[f"fundamental_revision_proxy_{n}d"] = roe_revision.stack()

    # --- 4. Liquidity stress proxy: bid-ask-spread-style estimate from high-low range (6 features) ---
    # Real bid-ask spread data requires tick-level feeds. As a proxy, we use the
    # Corwin-Schultz style high-low spread estimator, a documented academic
    # approximation of effective spread from daily OHLC data alone.
    beta_term = (np.log(high / low)) ** 2
    for n in [10, 20, 40, 60, 90, 120]:
        beta_sum = beta_term.rolling(2).sum()
        gamma_term = (np.log(high.rolling(2).max() / low.rolling(2).min())) ** 2
        alpha = (
            (np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)) / (3 - 2 * np.sqrt(2))
            - np.sqrt(gamma_term / (3 - 2 * np.sqrt(2)))
        )
        spread_est = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        feats[f"spread_proxy_{n}d"] = spread_est.rolling(n).mean().stack()

    # --- 5. Sentiment-style proxy: price acceleration relative to volume trend (6 features) ---
    # Real sentiment data comes from news/social feeds. As a proxy, we combine
    # price momentum with volume trend direction as a crude "attention" signal.
    for n in [20, 40, 60, 90, 120, 180]:
        price_accel = close.pct_change(n) - close.pct_change(n * 2)
        volume_trend = volume.rolling(n).mean() / volume.rolling(n * 2).mean()
        feats[f"attention_proxy_{n}d"] = (price_accel * volume_trend).stack()

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

    print("\nComputing alternative (proxy) features...")
    print("NOTE: these are derived proxies, not real short interest / options / sentiment data.")
    feats = compute_alternative_features(prices, fundamentals_aligned)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of alternative features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 30, f"Expected 30 alternative features, got {feats.shape[1]}"
    print("\nPASS: exactly 30 alternative features generated")