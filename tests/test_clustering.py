"""Tests for clustering algorithm (Stage B).

Covers the four required cases from CLAUDE.md §5:
(a) two spatially-separate clusters near-coincident in time
(b) one cluster with a late low-ToT pixel (time-walk)
(c) out-of-order pixel arrival
(d) clusters spanning a close-out boundary
"""

import numpy as np
import pandas as pd
import pytest

from tpx3pipe.clustering import cluster_pixels


def _make_pixel_df(rows):
    """rows: list of (x, y, time_ns, tot)"""
    return pd.DataFrame(rows, columns=["x_pix", "y_pix", "time_ns", "tot"])


# (a) Two spatially separate clusters, coincident in time
def test_spatially_separate_clusters():
    rows = [
        # cluster A around (0, 0)
        (0, 0, 100.0, 200),
        (1, 0, 110.0, 200),
        # cluster B around (100, 100) — far away, same time window
        (100, 100, 105.0, 200),
        (101, 100, 115.0, 200),
    ]
    df = _make_pixel_df(rows)
    clusters, _ = cluster_pixels(df, dt_ns=2000, max_gap=0)
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"


# (b) Cluster with a late low-ToT pixel (time-walk emulation)
def test_late_low_tot_pixel_same_cluster():
    rows = [
        (10, 10, 1000.0, 500),
        (11, 10, 1010.0, 480),
        (10, 11, 1950.0, 30),   # low ToT, arrives ~950 ns later — still within Δt=2000
    ]
    df = _make_pixel_df(rows)
    clusters, pix_out = cluster_pixels(df, dt_ns=2000, max_gap=0)
    assert len(clusters) == 1, f"Time-walk pixel should be in same cluster, got {len(clusters)}"
    assert clusters.iloc[0]["n_pixels"] == 3


def test_late_pixel_splits_cluster():
    """Pixel arriving > Δt after the cluster should start a new cluster."""
    rows = [
        (10, 10, 1000.0, 500),
        (11, 10, 3500.0, 30),   # 2500 ns later — exceeds Δt=2000 → new cluster
    ]
    df = _make_pixel_df(rows)
    clusters, _ = cluster_pixels(df, dt_ns=2000, max_gap=0)
    assert len(clusters) == 2, "Pixel beyond Δt should start a new cluster"


# (c) Out-of-order pixel arrival
def test_out_of_order_pixels():
    """Pixels arrive shuffled in time but belong to one cluster."""
    rows = [
        (5, 5, 300.0, 200),
        (5, 6, 100.0, 200),   # earlier in real time, arrives second
        (6, 5, 200.0, 200),
    ]
    df = _make_pixel_df(rows)
    clusters, _ = cluster_pixels(df, dt_ns=2000, max_gap=0)
    assert len(clusters) == 1, "Out-of-order pixels should be merged into one cluster"


# (d) Clusters spanning a close-out boundary
def test_closeout_boundary():
    """Two clusters separated by more than Δt + safety_margin in time."""
    dt = 500.0
    safety = 100.0
    rows = [
        (0, 0, 0.0, 200),
        (1, 0, 100.0, 200),
        # gap > dt + safety → next cluster starts after close-out
        (0, 0, 800.0, 200),  # 800 - 100 = 700 > dt=500 → new cluster
        (1, 0, 900.0, 200),
    ]
    df = _make_pixel_df(rows)
    clusters, _ = cluster_pixels(df, dt_ns=dt, max_gap=0, safety_margin_ns=safety)
    assert len(clusters) == 2, f"Expected 2 clusters across boundary, got {len(clusters)}"


# gap-tolerant connectivity
def test_gap_connectivity():
    """With max_gap=1, pixels at Chebyshev distance 2 should merge."""
    rows = [
        (0, 0, 100.0, 200),
        (2, 0, 110.0, 200),   # distance 2 — merged with max_gap=1
    ]
    df = _make_pixel_df(rows)
    clusters_strict, _ = cluster_pixels(df, dt_ns=2000, max_gap=0)
    clusters_gap, _ = cluster_pixels(df, dt_ns=2000, max_gap=1)
    assert len(clusters_strict) == 2
    assert len(clusters_gap) == 1


# cluster_id assignment completeness
def test_all_pixels_assigned(synth_data):
    pixel_df, _, _ = synth_data
    clusters, cluster_pix = cluster_pixels(pixel_df, dt_ns=2000, max_gap=0)
    assert "cluster_id" in cluster_pix.columns
    assert cluster_pix["cluster_id"].notna().all()
    assert len(cluster_pix) == len(pixel_df)
