import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def compute_composite_features(
    momentum_feats: pd.DataFrame,
    value_feats: pd.DataFrame,
    quality_feats: pd.DataFrame,
    volatility_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute composite factor features by combining signals across categories.
    Inputs: the individual feature DataFrames from momentum.py, value.py,
      quality.py, volatility.py — all indexed by (date, ticker).
    Returns: DataFrame indexed by (date, ticker) with 12 composite feature columns.
    """
    feats = pd.DataFrame(index=momentum_feats.index)

    def cs_rank(series: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank, date by date."""
        return series.groupby(level="date").rank(pct=True)

    # --- 1. Multi-factor blended scores (4 features) ---
    mom_score = cs_rank(momentum_feats["mom_12_1"])
    value_score = cs_rank(value_feats["value_composite_raw"])
    quality_score = cs_rank(quality_feats["profitability_composite"])
    low_vol_score = 1.0 - cs_rank(volatility_feats["realized_vol_60d"])

    feats["composite_mom_value"] = (mom_score + value_score) / 2
    feats["composite_quality_value"] = (quality_score + value_score) / 2
    feats["composite_mom_quality_lowvol"] = (mom_score + quality_score + low_vol_score) / 3
    feats["composite_all_four"] = (mom_score + value_score + quality_score + low_vol_score) / 4

    # --- 2. Risk-adjusted composite: blended score divided by volatility (2 features) ---
    combined_raw = mom_score + value_score + quality_score
    feats["composite_risk_adjusted"] = combined_raw / (1 + cs_rank(volatility_feats["realized_vol_60d"]))
    feats["composite_quality_risk_adjusted"] = quality_score / (1 + cs_rank(volatility_feats["realized_vol_60d"]))

    # --- 3. PCA-derived factors from a broad feature blend (4 features) ---
    # Combine a representative subset of columns from each category, then extract
    # the top 4 principal components as compressed "meta-factors."
    pca_input_cols = pd.concat([
        momentum_feats[["mom_12_1", "mom_6_1", "risk_adj_mom_60d"]],
        value_feats[["pe_zscore", "pb_zscore", "earnings_yield_zscore"]],
        quality_feats[["roe_zscore", "roa_zscore", "debt_equity_zscore"]],
        volatility_feats[["realized_vol_60d", "downside_dev_60d", "beta_252d"]],
    ], axis=1)

    pca_input_clean = pca_input_cols.dropna()
    if len(pca_input_clean) > 50:  # need enough rows for PCA to be meaningful
        pca = PCA(n_components=4)
        pca_values = pca.fit_transform(pca_input_clean.values)
        pca_df = pd.DataFrame(
            pca_values,
            index=pca_input_clean.index,
            columns=[f"pca_factor_{i+1}" for i in range(4)],
        )
        pca_df_full = pca_df.reindex(feats.index)
        for col in pca_df_full.columns:
            feats[col] = pca_df_full[col]
    else:
        for i in range(4):
            feats[f"pca_factor_{i+1}"] = np.nan

    # --- 4. Consensus signal: agreement across categories (2 features) ---
    signals_agree = pd.concat([
        (mom_score > 0.5).astype(int),
        (value_score > 0.5).astype(int),
        (quality_score > 0.5).astype(int),
        (low_vol_score > 0.5).astype(int),
    ], axis=1)
    feats["consensus_bullish_count"] = signals_agree.sum(axis=1)
    feats["consensus_agreement_pct"] = signals_agree.mean(axis=1)

    return feats


if __name__ == "__main__":
    from agents.signal_generation.data_loader import (
        load_ohlcv, load_fundamentals, align_fundamentals_to_prices,
    )
    from agents.signal_generation.features.momentum import compute_momentum_features
    from agents.signal_generation.features.value import compute_value_features
    from agents.signal_generation.features.quality import compute_quality_features
    from agents.signal_generation.features.volatility import compute_volatility_features

    print("Loading sample data...")
    prices = load_ohlcv("data/processed/sample_prices.parquet")
    fundamentals = load_fundamentals("data/processed/sample_fundamentals.parquet")

    price_dates = prices.index.get_level_values("date").unique().sort_values()
    tickers = prices.index.get_level_values("ticker").unique().tolist()
    fundamentals_aligned = align_fundamentals_to_prices(fundamentals, price_dates, tickers)

    print("Computing upstream feature categories (momentum, value, quality, volatility)...")
    mom_feats = compute_momentum_features(prices)
    val_feats = compute_value_features(prices, fundamentals_aligned)
    qual_feats = compute_quality_features(prices, fundamentals_aligned)
    vol_feats = compute_volatility_features(prices)

    print("\nComputing composite features...")
    feats = compute_composite_features(mom_feats, val_feats, qual_feats, vol_feats)

    print(f"\nFeature matrix shape: {feats.shape}")
    print(f"Number of composite features generated: {feats.shape[1]}")
    print(f"\nFeature columns:\n{list(feats.columns)}")

    print(f"\nSample (last 5 rows):")
    print(feats.tail())

    nan_frac = feats.isna().mean().mean()
    print(f"\nAverage NaN fraction across all features: {nan_frac:.2%}")

    assert feats.shape[1] == 12, f"Expected 12 composite features, got {feats.shape[1]}"
    print("\nPASS: exactly 12 composite features generated")

    # --- Full pipeline total check ---
    total_features = (
        mom_feats.shape[1] + val_feats.shape[1] + qual_feats.shape[1]
        + vol_feats.shape[1] + feats.shape[1]
    )
    print(f"\nRunning total (momentum + value + quality + volatility + composite): {total_features}")
    print("(Full 312 total will be confirmed once volume/technical/alternative are combined in the next milestone)")