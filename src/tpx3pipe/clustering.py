"""Stage B — Streaming open-cluster algorithm (Meduna et al., CTD/WIT 2019).

Key optimisations over the naive implementation:
- 256×256 numpy grid for spatial lookups (array indexing) instead of a
  Python dict keyed by (x, y) tuples.
- Pre-allocated numpy arrays for UF parent/rank and cluster metadata instead
  of Python dicts (avoids per-element object overhead).
- Lazy cluster close: stale grid entries are ignored on encounter; closing a
  cluster only removes it from a small Python set — the grid is never scanned.
- Vectorised post-processing: UF root resolution via numpy remap arrays;
  cluster summary via pandas groupby instead of per-cluster boolean masks.

These eliminate the two dominant bottlenecks in the earlier implementation:
  1. O(n_active_pixels) stale-entry scan on every cluster close.
  2. O(n_clusters × n_pixels) boolean-mask loop for the cluster summary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def cluster_pixels(
    pixel_df: pd.DataFrame,
    dt_ns: float = 2000.0,
    max_gap: int = 0,
    time_criterion: str = "chain",
    safety_margin_ns: float = 200.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster pixel hits using a streaming open-cluster algorithm.

    Returns
    -------
    clusters_df       : one row per cluster (cluster_id, n_pixels, t_min, t_max)
    cluster_pixels_df : pixel_df with added 'cluster_id' column
    """
    df = pixel_df.sort_values("time_ns").reset_index(drop=True)
    n = len(df)
    if n == 0:
        empty_cl = pd.DataFrame(columns=["cluster_id", "n_pixels", "t_min", "t_max"])
        return empty_cl, df

    times = df["time_ns"].to_numpy(dtype=np.float64)
    xs    = df["x_pix"].to_numpy(dtype=np.int32)
    ys    = df["y_pix"].to_numpy(dtype=np.int32)

    d          = 1 + max_gap          # Chebyshev radius
    close_lag  = dt_ns + safety_margin_ns
    use_chain  = (time_criterion == "chain")

    # ------------------------------------------------------------------
    # Pre-allocated arrays (worst case: every pixel is its own cluster)
    # ------------------------------------------------------------------
    uf_parent = np.arange(n, dtype=np.int32)   # UF parent
    uf_rank   = np.zeros(n, dtype=np.int8)     # UF rank

    cl_t_min  = np.empty(n, dtype=np.float64); cl_t_min[:] =  np.inf
    cl_t_max  = np.empty(n, dtype=np.float64); cl_t_max[:] = -np.inf
    cl_open   = np.zeros(n, dtype=bool)        # is this root currently open?

    # Spatial hash: 256×256 grid → most recent cluster root at that pixel
    # Stale entries (pointing to closed/merged roots) are handled lazily.
    grid = np.full((256, 256), -1, dtype=np.int32)

    pixel_cids = np.full(n, -1, dtype=np.int32)  # pixel → initial cluster id
    next_id    = 0
    open_set: set = set()   # roots that are currently open

    # ------------------------------------------------------------------
    # Inline union-find helpers (closures over numpy arrays)
    # ------------------------------------------------------------------
    def _find(x: int) -> int:
        p = uf_parent
        while p[x] != x:
            p[x] = p[p[x]]    # path halving
            x = int(p[x])
        return x

    def _union(a: int, b: int) -> int:
        ra = _find(a)
        rb = _find(b)
        if ra == rb:
            return ra
        if uf_rank[ra] < uf_rank[rb]:
            ra, rb = rb, ra
        uf_parent[rb] = ra
        if uf_rank[ra] == uf_rank[rb]:
            uf_rank[ra] += 1
        # propagate metadata into new root
        if cl_t_min[rb] < cl_t_min[ra]:
            cl_t_min[ra] = cl_t_min[rb]
        if cl_t_max[rb] > cl_t_max[ra]:
            cl_t_max[ra] = cl_t_max[rb]
        if cl_open[rb]:
            cl_open[ra] = True
            open_set.discard(rb)
            open_set.add(ra)
        cl_open[rb] = False
        return ra

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    log_step = max(1_000_000, n // 10)

    for i in range(n):
        t  = times[i]
        xi = int(xs[i])
        yi = int(ys[i])

        # Close stale clusters — only scans open_set (typically < 10 entries).
        # Grid entries pointing to closed roots are left in place and skipped lazily.
        if open_set:
            dead = [r for r in open_set if cl_t_max[r] + close_lag < t]
            if dead:
                for r in dead:
                    cl_open[r] = False
                    open_set.discard(r)

        # Neighbour search: slice a (2d+1)×(2d+1) patch from the grid.
        x0 = xi - d if xi > d else 0
        x1 = xi + d + 1 if xi + d < 256 else 256
        y0 = yi - d if yi > d else 0
        y1 = yi + d + 1 if yi + d < 256 else 256

        nb_roots: set = set()
        for raw in grid[x0:x1, y0:y1].flat:
            if raw < 0:
                continue
            r = _find(raw)
            if not cl_open[r]:
                continue                       # lazily skip closed/stale
            if use_chain:
                if t - cl_t_max[r] <= dt_ns:
                    nb_roots.add(r)
            else:
                if t - cl_t_min[r] <= dt_ns:
                    nb_roots.add(r)

        if not nb_roots:
            # start a new cluster
            cid = next_id; next_id += 1
            cl_t_min[cid] = t
            cl_t_max[cid] = t
            cl_open[cid]  = True
            open_set.add(cid)
            grid[xi, yi]  = cid
            pixel_cids[i] = cid
        else:
            # join and merge all neighbouring clusters
            nb_list = list(nb_roots)
            root = nb_list[0]
            for other in nb_list[1:]:
                root = _union(root, other)
            root = _find(root)

            if cl_t_min[root] > t: cl_t_min[root] = t
            if cl_t_max[root] < t: cl_t_max[root] = t
            cl_open[root] = True
            open_set.add(root)

            grid[xi, yi]  = root      # keep grid up-to-date with current root
            pixel_cids[i] = root

        if i % log_step == 0 and i > 0:
            log.info("  clustered %d / %d pixels  (%d open clusters)",
                     i, n, len(open_set))

    log.info("  clustered %d / %d pixels  (%d open clusters)", n, n, len(open_set))

    # ------------------------------------------------------------------
    # Vectorised post-processing
    # ------------------------------------------------------------------
    # 1. Resolve all pixel cluster IDs through UF (work only on unique IDs)
    raw_unique = np.unique(pixel_cids[pixel_cids >= 0])
    root_of    = np.empty(len(raw_unique), dtype=np.int32)
    for j, rid in enumerate(raw_unique):
        root_of[j] = _find(int(rid))

    # Build remap array: raw_id → root_id
    if len(raw_unique) > 0:
        remap = np.full(int(pixel_cids.max()) + 1, -1, dtype=np.int32)
        remap[raw_unique] = root_of
        resolved = np.where(pixel_cids >= 0, remap[pixel_cids], -1)
    else:
        resolved = pixel_cids.copy()

    # 2. Compact root IDs to 0-based contiguous integers
    unique_roots = np.unique(resolved[resolved >= 0])
    if len(unique_roots) > 0:
        compact = np.full(int(unique_roots.max()) + 2, -1, dtype=np.int32)
        compact[unique_roots] = np.arange(len(unique_roots), dtype=np.int32)
        cluster_id_col = np.where(resolved >= 0, compact[resolved], -1)
    else:
        cluster_id_col = np.full(n, -1, dtype=np.int32)

    df = df.copy()
    df["cluster_id"] = cluster_id_col

    # 3. Cluster summary via groupby — O(n), not O(n_clusters × n_pixels)
    labeled = df[cluster_id_col >= 0]
    if len(labeled) > 0:
        agg = labeled.groupby("cluster_id")["time_ns"].agg(["count", "min", "max"])
        clusters_df = pd.DataFrame({
            "cluster_id": agg.index.to_numpy(dtype=np.int32),
            "n_pixels":   agg["count"].to_numpy(dtype=np.int32),
            "t_min":      agg["min"].to_numpy(dtype=np.float64),
            "t_max":      agg["max"].to_numpy(dtype=np.float64),
        })
    else:
        clusters_df = pd.DataFrame(columns=["cluster_id", "n_pixels", "t_min", "t_max"])

    log.info(
        "Clustering complete: %d pixels → %d clusters (dt=%.0f ns, max_gap=%d)",
        n, len(clusters_df), dt_ns, max_gap,
    )
    return clusters_df, df


# ---------------------------------------------------------------------------
# Parameter-sensitivity sweep
# ---------------------------------------------------------------------------

def sweep(
    pixel_df: pd.DataFrame,
    dt_values: List[float],
    max_gap_values: List[int],
    plots_dir: Path,
) -> pd.DataFrame:
    """Sweep (dt_ns, max_gap) and return a summary DataFrame."""
    import matplotlib.pyplot as plt

    rows = []
    for dt in dt_values:
        for mg in max_gap_values:
            cl, _ = cluster_pixels(pixel_df, dt_ns=dt, max_gap=mg)
            sizes = cl["n_pixels"].values
            rows.append({
                "dt_ns":       dt,
                "max_gap":     mg,
                "n_clusters":  len(cl),
                "mean_size":   float(sizes.mean()),
                "median_size": float(np.median(sizes)),
                "p95_size":    float(np.percentile(sizes, 95)),
            })

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mg in max_gap_values:
        sub = summary[summary["max_gap"] == mg]
        axes[0].plot(sub["dt_ns"], sub["n_clusters"],  marker="o", label=f"gap={mg}")
        axes[1].plot(sub["dt_ns"], sub["mean_size"],   marker="o", label=f"gap={mg}")
        axes[2].plot(sub["dt_ns"], sub["p95_size"],    marker="o", label=f"gap={mg}")
    for ax, ylabel in zip(axes, ["# clusters", "mean size (px)", "p95 size (px)"]):
        ax.set_xlabel("dt (ns)"); ax.set_ylabel(ylabel)
        ax.legend(); ax.set_title(ylabel + " vs dt")
    fig.suptitle("Clustering parameter sweep")
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "clustering_sweep.png", dpi=120)
    plt.close(fig)
    log.info("Sweep plot saved to %s", plots_dir / "clustering_sweep.png")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage B: cluster pixel hits")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, load_pixels, ensure_artifacts_dir, ensure_plots_dir
    cfg      = load_config(args.config)
    pixel_df = load_pixels(cfg["paths"]["pixel_data"])
    out_dir  = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    ccfg = cfg.get("clustering", {})
    clusters_df, cluster_pixels_df = cluster_pixels(
        pixel_df,
        dt_ns            = ccfg.get("dt_ns", 2000),
        max_gap          = ccfg.get("max_gap", 0),
        time_criterion   = ccfg.get("time_criterion", "chain"),
        safety_margin_ns = ccfg.get("safety_margin_ns", 200),
    )

    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    cluster_pixels_df.to_parquet(out_dir / "cluster_pixels.parquet", index=False)
    print(f"Wrote {len(clusters_df)} clusters → {out_dir}/clusters.parquet")

    if args.sweep:
        sweep(pixel_df, dt_values=[200, 500, 1000, 2000],
              max_gap_values=[0, 1], plots_dir=plots_dir)


if __name__ == "__main__":
    main()
