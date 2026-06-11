"""Tests for contamination diagnostics (Stage G)."""

import numpy as np
import pandas as pd
import pytest

from tpx3pipe.diagnostics import baseline_under_peak, validate_against_truth


def _make_labeled_df(period=1785.0, seed=42):
    rng = np.random.default_rng(seed)
    n = 500
    ftof = np.concatenate([
        rng.normal(175.0, 15.0, 200),
        rng.normal(250.0, 20.0, 150),
        rng.uniform(0, period, 150),
    ])
    ftof = np.clip(ftof, 0, period)
    labels = np.concatenate([np.zeros(200), np.ones(150), np.full(150, 2)])
    return pd.DataFrame({
        "cluster_id": np.arange(n),
        "folded_tof_ns": ftof,
        "hard_label": labels,
    })


def test_baseline_under_peak_runs():
    df = _make_labeled_df()
    result = baseline_under_peak(df, 1785.0, pre_peak_window_ns=(0.0, 100.0))
    assert "background_density_per_ns" in result
    assert result["background_density_per_ns"] >= 0


def test_baseline_under_peak_contamination_in_range():
    df = _make_labeled_df()
    result = baseline_under_peak(df, 1785.0)
    for key in result:
        if "mislabel_frac" in key:
            assert 0.0 <= result[key] <= 1.0


def test_validate_against_truth_reports_error():
    df = _make_labeled_df()
    truth_df = pd.DataFrame({
        "cluster_id": df["cluster_id"],
        "wrap_count": np.concatenate([
            np.zeros(350, dtype=int),
            np.ones(150, dtype=int),
        ]),
    })
    estimated = {"class_0_mislabel_frac": 0.2}
    result = validate_against_truth(df, truth_df, estimated)
    assert "true_contamination_frac" in result
    assert "estimated_contamination_frac" in result
    assert "error" in result
    # true contamination should be 150/500 = 0.3
    assert abs(result["true_contamination_frac"] - 0.3) < 0.01
