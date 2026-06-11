"""Tests for synthetic data generator (Stage E)."""

import numpy as np
import pandas as pd
import pytest


def test_synth_schema(synth_data):
    pixel_df, tdc_df, truth_df = synth_data
    for col in ["x_pix", "y_pix", "toa", "ftoa", "spidr_time", "time_ns", "tot"]:
        assert col in pixel_df.columns, f"Missing pixel column: {col}"
    assert "time_ns" in tdc_df.columns
    for col in ["cluster_id", "species", "true_tof_ns", "wrap_count"]:
        assert col in truth_df.columns


def test_synth_pixel_bounds(synth_data):
    pixel_df, _, _ = synth_data
    assert (pixel_df["x_pix"] >= 0).all() and (pixel_df["x_pix"] <= 255).all()
    assert (pixel_df["y_pix"] >= 0).all() and (pixel_df["y_pix"] <= 255).all()


def test_synth_tdc_monotone(synth_data):
    _, tdc_df, _ = synth_data
    diffs = tdc_df["time_ns"].diff().dropna()
    assert (diffs > 0).all(), "TDC edges are not monotonically increasing"


def test_synth_wraparound_present(synth_data, base_cfg):
    _, _, truth_df = synth_data
    period = base_cfg["synth"]["period_ns"]
    wrapped = truth_df[truth_df["species"] == "slow"]
    if len(wrapped) > 0:
        # slow species has tof_mean > period → most should wrap
        assert (wrapped["wrap_count"] > 0).mean() > 0.5, \
            "Expected >50% of 'slow' clusters to have wrap_count > 0"


def test_synth_n_clusters(synth_data):
    _, _, truth_df = synth_data
    assert len(truth_df) > 10, "Expected at least 10 clusters in 0.5 s run"
