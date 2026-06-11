"""Tests for time-block split utility (no temporal overlap)."""

import numpy as np
import pandas as pd
import pytest

from tpx3pipe.baseline import time_block_split


def _make_df(n=100):
    return pd.DataFrame({
        "cluster_id": np.arange(n),
        "t_min": np.linspace(0, 100_000, n),
        "hard_label": np.random.randint(0, 3, n),
    })


def test_no_temporal_overlap():
    df = _make_df(200)
    train, val, test = time_block_split(df, train_frac=0.70, val_frac=0.15)
    assert train["t_min"].max() <= val["t_min"].min()
    assert val["t_min"].max() <= test["t_min"].min()


def test_split_sizes():
    n = 100
    df = _make_df(n)
    train, val, test = time_block_split(df, train_frac=0.70, val_frac=0.15)
    assert len(train) + len(val) + len(test) == n


def test_split_covers_all():
    df = _make_df(100)
    train, val, test = time_block_split(df)
    all_ids = set(train["cluster_id"]) | set(val["cluster_id"]) | set(test["cluster_id"])
    assert all_ids == set(df["cluster_id"])


def test_random_split_forbidden():
    """Verify that time_block_split sorts by time, not randomly."""
    df = _make_df(100)
    train1, _, _ = time_block_split(df)
    train2, _, _ = time_block_split(df)
    # deterministic result
    pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                   train2.reset_index(drop=True))
