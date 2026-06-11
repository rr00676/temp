"""Performance test for the clustering algorithm.

Generates a realistic synthetic dataset (~1M pixels) and verifies that
clustering completes within a time budget.  Run with -s to see timing.
"""

import time
import numpy as np
import pandas as pd
import pytest

from tpx3pipe.clustering import cluster_pixels


def _make_large_pixel_df(n_pixels: int, period_ns: float = 1785.0,
                          duration_s: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Generate a realistic Timepix3-like pixel stream.

    Produces random clusters with ~5 pixels each, distributed across the
    256×256 detector area, spanning `duration_s` seconds.
    """
    rng = np.random.default_rng(seed)
    n_clusters = n_pixels // 5

    # Cluster arrival times uniformly distributed over duration
    t_cluster = rng.uniform(0, duration_s * 1e9, size=n_clusters)

    rows = []
    for t0 in t_cluster:
        size = int(rng.integers(2, 9))
        cx   = int(rng.integers(2, 254))
        cy   = int(rng.integers(2, 254))
        for _ in range(size):
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))
            dt = rng.exponential(200.0)
            rows.append((
                max(0, min(255, cx + dx)),
                max(0, min(255, cy + dy)),
                t0 + dt,
                int(rng.integers(50, 600)),
            ))

    df = pd.DataFrame(rows, columns=["x_pix", "y_pix", "time_ns", "tot"])
    # Shuffle to simulate unsorted arrival
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# -----------------------------------------------------------------------
# Performance tests
# -----------------------------------------------------------------------

@pytest.mark.parametrize("n_pixels,budget_s", [
    (100_000,  5.0),    # 100k pixels — should finish in < 5 s
    (500_000, 25.0),    # 500k pixels — should finish in < 25 s
])
def test_clustering_throughput(n_pixels, budget_s):
    df = _make_large_pixel_df(n_pixels)
    t0 = time.perf_counter()
    clusters, pix_out = cluster_pixels(df, dt_ns=2000, max_gap=0)
    elapsed = time.perf_counter() - t0

    print(f"\n  {n_pixels:,} pixels -> {len(clusters):,} clusters in {elapsed:.2f} s "
          f"({n_pixels / elapsed / 1e3:.0f} k px/s)")

    assert elapsed < budget_s, (
        f"Clustering {n_pixels:,} pixels took {elapsed:.1f} s, budget is {budget_s} s"
    )
    # Correctness sanity: every pixel is assigned
    assert len(pix_out) == len(df)
    assert (pix_out["cluster_id"] >= 0).all()


def test_clustering_1m_pixels():
    """Soft benchmark: 1M pixels, no hard budget — just report throughput."""
    n = 1_000_000
    df = _make_large_pixel_df(n)
    t0 = time.perf_counter()
    clusters, _ = cluster_pixels(df, dt_ns=2000, max_gap=0)
    elapsed = time.perf_counter() - t0
    rate = n / elapsed / 1e3
    print(f"\n  1M pixels -> {len(clusters):,} clusters in {elapsed:.2f} s  ({rate:.0f} k px/s)")
    # Sanity only, no time budget
    assert len(clusters) > 0
