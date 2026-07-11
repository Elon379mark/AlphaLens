import pandas as pd
from pathlib import Path

from agents.signal_generation.data_loader import (
    load_ohlcv, load_fundamentals, align_fundamentals_to_prices,
)
from agents.signal_generation.features.momentum import compute_momentum_features
from agents.signal_generation.features.value import compute_value_features
from agents.signal_generation.features.quality import compute_quality_features
from agents.signal_generation.features.volatility import compute_volatility_features
from agents.signal_generation.features.volume import compute_volume_features
from agents.signal_generation.features.technical import compute_technical_features
from agents.signal_generation.features.alternative import compute_alternative_features
from agents.signal_generation.features.composite import compute_composite_features


EXPECTED_COUNTS = {
    "momentum": 60,
    "value": 50,
    "quality": 45,
    "volatility": 40,
    "volume": 35,
    "technical": 40,
    "alternative": 30,
    "composite": 12,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())  # 312


def build_all_features(
    ohlcv_path: str = "data/processed/sample_prices.parquet",
    fundamentals_path: str = "data/processed/sample_fundamentals.parquet",
) -> pd.DataFrame:
    """
    Run all 8 feature category modules and concatenate into one matrix.
    Returns a DataFrame indexed by (date, ticker) with all 312 feature columns.
    """
    print("Loading OHLCV and fundamentals...")
    prices = load_ohlcv(ohlcv_path)
    fundamentals = load_fundamentals(fundamentals_path)

    price_dates = prices.index.get_level_values("date").unique().sort_values()
    tickers = prices.index.get_level_values("ticker").unique().tolist()
    fundamentals_aligned = align_fundamentals_to_prices(fundamentals, price_dates, tickers)

    print("Computing momentum features (60)...")
    mom_feats = compute_momentum_features(prices)
    _check_count("momentum", mom_feats)

    print("Computing value features (50)...")
    val_feats = compute_value_features(prices, fundamentals_aligned)
    _check_count("value", val_feats)

    print("Computing quality features (45)...")
    qual_feats = compute_quality_features(prices, fundamentals_aligned)
    _check_count("quality", qual_feats)

    print("Computing volatility features (40)...")
    vola_feats = compute_volatility_features(prices)
    _check_count("volatility", vola_feats)

    print("Computing volume features (35)...")
    volu_feats = compute_volume_features(prices)
    _check_count("volume", volu_feats)

    print("Computing technical features (40)...")
    tech_feats = compute_technical_features(prices)
    _check_count("technical", tech_feats)

    print("Computing alternative features (30)...")
    alt_feats = compute_alternative_features(prices, fundamentals_aligned)
    _check_count("alternative", alt_feats)

    print("Computing composite features (12)...")
    comp_feats = compute_composite_features(mom_feats, val_feats, qual_feats, vola_feats)
    _check_count("composite", comp_feats)

    print("\nConcatenating all feature categories...")
    all_features = pd.concat([
        mom_feats, val_feats, qual_feats, vola_feats,
        volu_feats, tech_feats, alt_feats, comp_feats,
    ], axis=1)

    print(f"Combined shape: {all_features.shape}")
    assert all_features.shape[1] == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} total features, got {all_features.shape[1]}"
    )

    # Check for duplicate column names across categories (would silently overwrite data)
    dupes = all_features.columns[all_features.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate column names found across feature categories: {dupes}")

    return all_features


def _check_count(name: str, df: pd.DataFrame) -> None:
    expected = EXPECTED_COUNTS[name]
    actual = df.shape[1]
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected} features, got {actual}")
    print(f"  -> {name}: {actual} features OK")


def save_features(features: pd.DataFrame, path: str = "data/processed/features.parquet") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    features.reset_index().to_parquet(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    features = build_all_features()

    print(f"\n{'='*50}")
    print(f"TOTAL FEATURES: {features.shape[1]} (expected {EXPECTED_TOTAL})")
    print(f"TOTAL ROWS: {features.shape[0]}")
    print(f"{'='*50}")

    save_features(features)

    print("\nPer-category breakdown:")
    for name, count in EXPECTED_COUNTS.items():
        print(f"  {name}: {count}")
    print(f"  TOTAL: {sum(EXPECTED_COUNTS.values())}")

    assert features.shape[1] == 312, f"Final check failed: got {features.shape[1]} features"
    print("\nPASS: full 312-feature matrix built and saved successfully")