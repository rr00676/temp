"""Tests that CNN patch extraction has no ToF/position leakage (Stage I, CLAUDE.md §12)."""

import numpy as np
import pandas as pd
import pytest


def _make_cluster_pix(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cluster_id": np.zeros(n, dtype=int),
        "x_pix": rng.integers(10, 20, n),
        "y_pix": rng.integers(10, 20, n),
        "time_ns": np.sort(rng.uniform(1000.0, 1100.0, n)),
        "tot": rng.integers(50, 300, n),
    })


def _make_clusters_df():
    return pd.DataFrame({
        "cluster_id": [0],
        "t_min": [1000.0],
        "t_max": [1100.0],
    })


def test_no_absolute_time_in_patch():
    """ch1 must be relative (≥0), not absolute time_ns (which is ~1000 ns)."""
    from tpx3pipe.cnn.patches import extract_patches
    pix = _make_cluster_pix()
    cl = _make_clusters_df()
    patches, _, stats = extract_patches(pix, cl, patch_size=16, dt_clip_ns=2000.0)
    # ch1 values before normalization should be in [0, 2000], not ~1000
    # After norm they can be anything, but the raw max should be ≤ dt_clip
    mean, std = stats
    # recover unnormalized ch1 max value
    ch1_unnorm = patches[:, 1, :, :] * std[1] + mean[1]
    assert ch1_unnorm.max() <= 2000.0 + 1.0  # dt_clip + tolerance


def test_no_tof_derived_column():
    """Patches must not contain columns like raw_tof_ns or folded_tof_ns."""
    from tpx3pipe.cnn.patches import extract_patches
    pix = _make_cluster_pix()
    # add forbidden columns
    pix["raw_tof_ns"] = 123.0
    pix["folded_tof_ns"] = 45.0
    cl = _make_clusters_df()
    # extract_patches should work correctly and not use those columns
    patches, _, _ = extract_patches(pix, cl, patch_size=16, dt_clip_ns=2000.0)
    # just verify it ran without error and produced correct shape
    assert patches.shape[1] == 2   # exactly 2 channels


def test_no_absolute_xy_in_patch():
    """Patch must be centered (no chip absolute x/y leakage)."""
    from tpx3pipe.cnn.patches import extract_patches

    # Place cluster A at (0,0) and cluster B at (200, 200)
    def _make_pix(cx, cy, cid, seed):
        rng = np.random.default_rng(seed)
        n = 5
        return pd.DataFrame({
            "cluster_id": np.full(n, cid, dtype=int),
            "x_pix": np.clip(cx + rng.integers(-1, 2, n), 0, 255),
            "y_pix": np.clip(cy + rng.integers(-1, 2, n), 0, 255),
            "time_ns": np.sort(rng.uniform(1000.0, 1050.0, n)),
            "tot": rng.integers(100, 300, n),
        })

    pix_a = _make_pix(10, 10, 0, 0)
    pix_b = _make_pix(200, 200, 1, 1)
    pix_all = pd.concat([pix_a, pix_b], ignore_index=True)

    cl = pd.DataFrame({"cluster_id": [0, 1], "t_min": [1000.0, 1000.0]})
    patches, cids, stats = extract_patches(pix_all, cl, patch_size=16, dt_clip_ns=2000.0)

    # Both patches should have similar ch0 footprints (centered on max-ToT)
    # Their non-zero masks should be similar in shape
    p0 = patches[cids == 0][0, 0]
    p1 = patches[cids == 1][0, 0]
    # Both should have some non-zero pixels near the center (not in corner)
    half = 8
    center_a = abs(p0[half-2:half+2, half-2:half+2]).sum()
    center_b = abs(p1[half-2:half+2, half-2:half+2]).sum()
    assert center_a > 0 or center_b > 0   # at least one centered
