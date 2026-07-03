"""
test_features.py
Unit tests for the AlphaLens Signal Generation feature factory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import pandas as pd
import numpy as np

from signal_generation.data_loader import create_sample_data
from signal_generation.features import compute_all_features
from signal_generation.features.momentum import compute_momentum_features
from signal_generation.features.value import compute_value_features
from signal_generation.features.quality import compute_quality_features
from signal_generation.features.volatility import compute_volatility_features
from signal_generation.features.volume import compute_volume_features
from signal_generation.features.technical import compute_technical_features
from signal_generation.features.alternative import compute_alternative_features
from signal_generation.features.composite import compute_composite_features


@pytest.fixture(scope="module")
def sample_data():
    ohlcv, fundamentals = create_sample_data(n_tickers=15, n_days=400, seed=7)
    return ohlcv, fundamentals


def test_momentum_features_shape(sample_data):
    ohlcv, _ = sample_data
    feats = compute_momentum_features(ohlcv)
    assert feats.shape[1] >= 50, f"Expected >=50 momentum features, got {feats.shape[1]}"
    assert isinstance(feats.index, pd.MultiIndex)
    assert feats.index.names == ["date", "ticker"]


def test_value_features_shape(sample_data):
    ohlcv, fundamentals = sample_data
    feats = compute_value_features(ohlcv, fundamentals)
    assert feats.shape[1] >= 30, f"Expected >=30 value features, got {feats.shape[1]}"


def test_quality_features_shape(sample_data):
    _, fundamentals = sample_data
    feats = compute_quality_features(fundamentals)
    assert feats.shape[1] >= 25, f"Expected >=25 quality features, got {feats.shape[1]}"


def test_volatility_features_shape(sample_data):
    ohlcv, _ = sample_data
    feats = compute_volatility_features(ohlcv)
    assert feats.shape[1] >= 25, f"Expected >=25 volatility features, got {feats.shape[1]}"


def test_volume_features_shape(sample_data):
    ohlcv, _ = sample_data
    feats = compute_volume_features(ohlcv)
    assert feats.shape[1] >= 25, f"Expected >=25 volume features, got {feats.shape[1]}"


def test_technical_features_shape(sample_data):
    ohlcv, _ = sample_data
    feats = compute_technical_features(ohlcv)
    assert feats.shape[1] >= 30, f"Expected >=30 technical features, got {feats.shape[1]}"


def test_alternative_features_shape(sample_data):
    ohlcv, fundamentals = sample_data
    feats = compute_alternative_features(ohlcv, fundamentals)
    assert feats.shape[1] >= 15, f"Expected >=15 alternative features, got {feats.shape[1]}"


def test_composite_features_requires_inputs(sample_data):
    ohlcv, fundamentals = sample_data
    all_feats = compute_all_features(ohlcv, fundamentals)
    comp = compute_composite_features(all_feats)
    assert isinstance(comp, pd.DataFrame)


def test_compute_all_features_total_count(sample_data):
    ohlcv, fundamentals = sample_data
    feats = compute_all_features(ohlcv, fundamentals)
    assert feats.shape[1] >= 250, f"Expected close to 312 features, got {feats.shape[1]}"


def test_compute_all_features_no_fundamentals(sample_data):
    ohlcv, _ = sample_data
    feats = compute_all_features(ohlcv, fundamentals=None)
    # Without fundamentals, only momentum/volatility/volume/technical run
    assert feats.shape[1] >= 100
    assert isinstance(feats.index, pd.MultiIndex)


def test_no_infinite_values(sample_data):
    ohlcv, fundamentals = sample_data
    feats = compute_all_features(ohlcv, fundamentals)
    numeric = feats.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any(), "Features contain infinite values"


def test_feature_index_alignment(sample_data):
    ohlcv, fundamentals = sample_data
    feats = compute_all_features(ohlcv, fundamentals)
    # All feature rows should correspond to (date, ticker) pairs present in ohlcv
    assert set(feats.index.get_level_values("ticker")).issubset(
        set(ohlcv.index.get_level_values("ticker"))
    )
