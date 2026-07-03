"""
test_ic.py
Unit tests for IC / ICIR computation — AlphaLens Signal Generation Agent.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import pandas as pd
import numpy as np

from signal_generation.data_loader import create_sample_data
from signal_generation.ic_calculator import (
    compute_forward_returns,
    compute_ic_series,
    compute_ic,
    compute_icir,
    compute_all_ic_icir,
)


@pytest.fixture(scope="module")
def sample_data():
    ohlcv, _ = create_sample_data(n_tickers=20, n_days=400, seed=11)
    return ohlcv


def test_compute_forward_returns_shape(sample_data):
    fwd = compute_forward_returns(sample_data, horizon=21)
    assert isinstance(fwd, pd.Series)
    assert fwd.name == "fwd_return"
    assert isinstance(fwd.index, pd.MultiIndex)


def test_compute_forward_returns_is_nan_at_tail(sample_data):
    fwd = compute_forward_returns(sample_data, horizon=21)
    # Last 21 days per ticker should be NaN (no forward data)
    last_date = sample_data.index.get_level_values("date").max()
    tail_vals = fwd.xs(last_date, level="date")
    assert tail_vals.isna().all()


def test_compute_ic_series_returns_float_per_date(sample_data):
    fwd = compute_forward_returns(sample_data)
    # Use a feature with real signal: forward return itself shifted (perfect IC=1 sanity check)
    feature = sample_data["returns"]
    ic_series = compute_ic_series(feature, fwd)
    assert isinstance(ic_series, pd.Series)
    # Should be roughly bounded in [-1, 1] (Spearman correlation), ignoring NaNs
    assert ic_series.dropna().between(-1.0001, 1.0001).all()


def test_compute_ic_returns_float(sample_data):
    fwd = compute_forward_returns(sample_data)
    feature = sample_data["returns"]
    ic = compute_ic(feature, fwd)
    assert isinstance(ic, float)
    assert not np.isnan(ic)


def test_compute_icir_returns_float(sample_data):
    fwd = compute_forward_returns(sample_data)
    feature = sample_data["returns"]
    icir = compute_icir(feature, fwd)
    assert isinstance(icir, float)


def test_compute_icir_handles_zero_std():
    # Constant feature -> IC series should be NaN or zero std -> ICIR = 0.0
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=30), [f"T{i}" for i in range(10)]],
        names=["date", "ticker"],
    )
    constant_feature = pd.Series(1.0, index=idx)
    fwd = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(idx)), index=idx)
    icir = compute_icir(constant_feature, fwd)
    assert icir == 0.0


def test_compute_ic_handles_insufficient_observations():
    # Fewer than min_obs tickers per date -> IC should be NaN for those dates -> mean=0.0 if all NaN
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), [f"T{i}" for i in range(3)]],
        names=["date", "ticker"],
    )
    feature = pd.Series(np.random.default_rng(2).normal(size=len(idx)), index=idx)
    fwd = pd.Series(np.random.default_rng(3).normal(size=len(idx)), index=idx)
    ic = compute_ic(feature, fwd)
    assert ic == 0.0  # all dates have <10 obs (min_obs default), so IC series is all NaN


def test_compute_all_ic_icir_dict_keys_match_columns():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=60), [f"T{i}" for i in range(15)]],
        names=["date", "ticker"],
    )
    rng = np.random.default_rng(5)
    features = pd.DataFrame({
        "feat_a": rng.normal(size=len(idx)),
        "feat_b": rng.normal(size=len(idx)),
    }, index=idx)
    fwd = pd.Series(rng.normal(size=len(idx)), index=idx, name="fwd_return")

    ic_dict, icir_dict = compute_all_ic_icir(features, fwd)
    assert set(ic_dict.keys()) == {"feat_a", "feat_b"}
    assert set(icir_dict.keys()) == {"feat_a", "feat_b"}
    assert all(isinstance(v, float) for v in ic_dict.values())
    assert all(isinstance(v, float) for v in icir_dict.values())


def test_compute_all_ic_icir_handles_bad_feature_gracefully():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=60), [f"T{i}" for i in range(15)]],
        names=["date", "ticker"],
    )
    rng = np.random.default_rng(6)
    features = pd.DataFrame({
        "good_feat": rng.normal(size=len(idx)),
        "all_nan_feat": np.nan,
    }, index=idx)
    fwd = pd.Series(rng.normal(size=len(idx)), index=idx, name="fwd_return")

    ic_dict, icir_dict = compute_all_ic_icir(features, fwd)
    assert ic_dict["all_nan_feat"] == 0.0
    assert icir_dict["all_nan_feat"] == 0.0
