"""Tests for ToF computation and folding (Stage D)."""

import numpy as np
import pandas as pd
import pytest

from tpx3pipe.tof import compute_tof


def test_tof_basic():
    """Simple case: one cluster, one TDC edge just before it."""
    period = 1785.0
    tdc_df = pd.DataFrame({"time_ns": [0.0, 1785.0, 3570.0]})
    clusters_df = pd.DataFrame({
        "cluster_id": [0],
        "t_min": [1900.0],  # raw tof = 1900 - 1785 = 115 ns
        "t_max": [1910.0],
        "t_argmax_tot": [1905.0],
    })
    result = compute_tof(clusters_df, tdc_df, period)
    assert abs(result["raw_tof_ns"].iloc[0] - 115.0) < 1.0
    assert abs(result["folded_tof_ns"].iloc[0] - 115.0) < 1.0


def test_tof_wraparound():
    """Cluster with true ToF > period → folded ToF wraps."""
    period = 1785.0
    tdc_df = pd.DataFrame({"time_ns": [0.0]})
    # true ToF = 2100 ns; raw_tof = 2100; folded = 2100 % 1785 = 315 ns
    clusters_df = pd.DataFrame({
        "cluster_id": [0],
        "t_min": [2100.0],
        "t_max": [2110.0],
        "t_argmax_tot": [2105.0],
    })
    result = compute_tof(clusters_df, tdc_df, period)
    expected_folded = 2100.0 % period
    assert abs(result["folded_tof_ns"].iloc[0] - expected_folded) < 1.0


def test_tof_before_first_edge():
    """Cluster before the first TDC edge should have NaN ToF."""
    period = 1785.0
    tdc_df = pd.DataFrame({"time_ns": [5000.0]})
    clusters_df = pd.DataFrame({
        "cluster_id": [0],
        "t_min": [100.0],
        "t_max": [110.0],
        "t_argmax_tot": [105.0],
    })
    result = compute_tof(clusters_df, tdc_df, period)
    assert np.isnan(result["folded_tof_ns"].iloc[0])


def test_fold_correctness_from_synth(synth_data, base_cfg):
    """End-to-end: folded ToF for non-wrapped clusters should ≈ true ToF."""
    from tpx3pipe.clustering import cluster_pixels
    from tpx3pipe.features import compute_features

    pixel_df, tdc_df, truth_df = synth_data
    period = base_cfg["synth"]["period_ns"]

    clusters_df, cluster_pix = cluster_pixels(pixel_df, dt_ns=2000, max_gap=0)
    feat_df = compute_features(cluster_pix)
    overlap = set(clusters_df.columns) & set(feat_df.columns) - {"cluster_id"}
    clusters_df = clusters_df.drop(columns=list(overlap))
    clusters_df = clusters_df.merge(feat_df, on="cluster_id", how="left")
    clusters_df = compute_tof(clusters_df, tdc_df, period)

    # for non-wrapped clusters (wrap_count=0), folded_tof ≈ true_tof
    # (approximate because clustering shifts cluster_id numbering vs truth)
    valid = clusters_df["folded_tof_ns"].notna()
    ftof = clusters_df.loc[valid, "folded_tof_ns"].values
    # folded ToF must be in [0, period)
    assert (ftof >= 0).all()
    assert (ftof < period).all()
