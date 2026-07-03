"""
test_ranking.py
Unit tests for validation and ranking — AlphaLens Signal Generation Agent.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import pandas as pd
import numpy as np
import json
import tempfile

from signal_generation.validator import (
    validate_features,
    check_feature_correlation,
    get_validation_summary,
    IC_THRESHOLD,
    ICIR_THRESHOLD,
    NAN_THRESHOLD,
)
from signal_generation.ranker import (
    rank_signals,
    get_top_signals,
    save_ranked_signals,
    build_ranking_report,
)


@pytest.fixture
def sample_features_and_scores():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=30), [f"T{i}" for i in range(10)]],
        names=["date", "ticker"],
    )
    rng = np.random.default_rng(42)
    features = pd.DataFrame({
        "strong_signal": rng.normal(size=len(idx)),
        "weak_signal": rng.normal(size=len(idx)) * 0.001,
        "noisy_signal": np.where(rng.random(len(idx)) < 0.5, np.nan, rng.normal(size=len(idx))),
    }, index=idx)

    ic_dict = {"strong_signal": 0.08, "weak_signal": 0.005, "noisy_signal": 0.03}
    icir_dict = {"strong_signal": 1.2, "weak_signal": 0.1, "noisy_signal": 0.8}
    return features, ic_dict, icir_dict


def test_validate_features_filters_weak_signal(sample_features_and_scores):
    features, ic_dict, icir_dict = sample_features_and_scores
    validated = validate_features(features, ic_dict, icir_dict)
    assert "strong_signal" in validated.columns
    assert "weak_signal" not in validated.columns  # fails IC and ICIR thresholds


def test_validate_features_filters_high_nan(sample_features_and_scores):
    features, ic_dict, icir_dict = sample_features_and_scores
    validated = validate_features(features, ic_dict, icir_dict)
    # noisy_signal has 50% NaN > 30% threshold -> should be excluded
    assert "noisy_signal" not in validated.columns


def test_validate_features_empty_when_all_fail():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["A", "B"]],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"junk": [0.0] * len(idx)}, index=idx)
    ic_dict = {"junk": 0.001}
    icir_dict = {"junk": 0.01}
    validated = validate_features(features, ic_dict, icir_dict)
    assert validated.shape[1] == 0


def test_check_feature_correlation_detects_redundancy():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    df = pd.DataFrame({
        "a": base,
        "b": base + rng.normal(scale=0.001, size=200),  # near-duplicate
        "c": rng.normal(size=200),  # independent
    })
    redundant = check_feature_correlation(df, max_corr=0.95)
    assert "b" in redundant or "a" in redundant


def test_get_validation_summary_sorted_by_icir(sample_features_and_scores):
    features, ic_dict, icir_dict = sample_features_and_scores
    summary = get_validation_summary(features, ic_dict, icir_dict)
    assert list(summary.index)[0] == "strong_signal"  # highest |ICIR|
    assert "passed" in summary.columns


def test_rank_signals_descending_by_abs_icir():
    icir_dict = {"a": 0.5, "b": -1.2, "c": 0.1}
    ranked = rank_signals(icir_dict)
    assert ranked == ["b", "a", "c"]


def test_get_top_signals_respects_top_n():
    ranked = ["a", "b", "c", "d", "e"]
    top = get_top_signals(ranked, top_n=3)
    assert top == ["a", "b", "c"]


def test_save_ranked_signals_writes_json():
    ranked = ["sig1", "sig2", "sig3"]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "outputs", "ranked_signals.json")
        save_ranked_signals(ranked, path=path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == ranked


def test_build_ranking_report_structure():
    ranked = ["a", "b"]
    ic_dict = {"a": 0.05, "b": -0.03}
    icir_dict = {"a": 1.1, "b": -0.9}
    report = build_ranking_report(ranked, ic_dict, icir_dict, top_n=2)
    assert len(report) == 2
    assert report[0]["rank"] == 1
    assert report[0]["signal_name"] == "a"
    assert "ic" in report[0] and "icir" in report[0]
