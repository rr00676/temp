"""Tests for mixture-fit labeling (Stage F)."""

import numpy as np
import pytest

from tpx3pipe.labeling import GaussianUniformMixture


def test_em_recovers_two_peaks():
    """EM should recover two Gaussian peaks from synthetic data."""
    rng = np.random.default_rng(123)
    period = 1785.0
    X = np.concatenate([
        rng.normal(175.0, 15.0, 300),
        rng.normal(250.0, 20.0, 200),
        rng.uniform(0, period, 50),   # background
    ])
    X = np.clip(X, 0, period)

    model = GaussianUniformMixture(
        n_peaks=2, T=period,
        means_init=[170.0, 260.0],
        stds_init=[20.0, 20.0],
        n_iter=200,
    )
    model.fit(X)

    means_sorted = np.sort(model.means_)
    assert abs(means_sorted[0] - 175.0) < 20.0, f"Peak 0 mean off: {means_sorted[0]}"
    assert abs(means_sorted[1] - 250.0) < 20.0, f"Peak 1 mean off: {means_sorted[1]}"


def test_em_posteriors_sum_to_one():
    """Posteriors should sum to 1 per event."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1785.0, 100)
    model = GaussianUniformMixture(2, 1785.0, [175.0, 250.0], [20.0, 20.0], n_iter=10)
    model.fit(X)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_em_weights_sum_to_one():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1785.0, 100)
    model = GaussianUniformMixture(2, 1785.0, [175.0, 250.0], [20.0, 20.0], n_iter=20)
    model.fit(X)
    assert abs(model.weights_.sum() - 1.0) < 1e-6


def test_label_purity_vs_truth(synth_data, base_cfg):
    """Mixture-fit label purity should be within 0.25 of the true purity on synthetic data."""
    from tpx3pipe.clustering import cluster_pixels
    from tpx3pipe.features import compute_features
    from tpx3pipe.tof import compute_tof
    from tpx3pipe.labeling import label_clusters

    pixel_df, tdc_df, truth_df = synth_data
    period = base_cfg["synth"]["period_ns"]

    clusters_df, cpix = cluster_pixels(pixel_df, dt_ns=2000, max_gap=0)
    feat_df = compute_features(cpix)
    overlap = set(clusters_df.columns) & set(feat_df.columns) - {"cluster_id"}
    clusters_df = clusters_df.drop(columns=list(overlap))
    clusters_df = clusters_df.merge(feat_df, on="cluster_id", how="left")
    clusters_df = compute_tof(clusters_df, tdc_df, period)
    labeled_df, model = label_clusters(clusters_df, period, base_cfg)

    # ground-truth contamination: fraction of clusters with wrap_count > 0
    # (can only check if cluster_id mapping aligns — approximate check via proportion)
    valid = labeled_df["hard_label"].notna() & (labeled_df["hard_label"] >= 0)
    frac_background = (labeled_df.loc[valid, "hard_label"] == model.n_peaks).mean()
    true_wrapped_rate = (truth_df["wrap_count"] > 0).mean()

    # loose tolerance: within 0.25
    assert abs(frac_background - true_wrapped_rate) < 0.25, (
        f"Background fraction {frac_background:.3f} too far from true {true_wrapped_rate:.3f}"
    )
