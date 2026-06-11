"""Stage B — Streaming open-cluster algorithm (Meduna et al., CTD/WIT 2019).

Adaptations:
- time_ns is the sole time axis (already corrected upstream).
- Input is sorted by time_ns (cheap offline).
- Temporal criterion: "chain" (within Δt of cluster's latest pixel) or
  "min" (within Δt of cluster's earliest pixel) — config option.
- Spatial criterion: Chebyshev distance ≤ 1 + max_gap (configurable).
- Cluster merge via union-find.
- A cluster is closed when the stream time exceeds
  cluster_max_time + Δt + safety_margin_ns.

Outputs (parquet):
- clusters.parquet     — one row per cluster (id, n_pixels, t_min, t_max, ...)
- cluster_pixels.parquet — pixel→cluster assignment

Usage::

    python -m tpx3pipe.clustering --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-find (path-compressed, union-by-rank)
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self._parent: Dict[int, int] = {}
        self._rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class _ClusterState:
    __slots__ = ("id", "t_min", "t_max", "t_ref", "pixel_indices")

    def __init__(self, cid: int, t: float, pix_idx: int, time_criterion: str):
        self.id = cid
        self.t_min = t
        self.t_max = t
        self.t_ref = t          # used for chain criterion (latest pixel time)
        self.pixel_indices: List[int] = [pix_idx]


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
    """Cluster pixel hits.

    Returns
    -------
    clusters_df : DataFrame with one row per cluster
    cluster_pixels_df : pixel_df with added 'cluster_id' column
    """
    # sort by time
    df = pixel_df.sort_values("time_ns").reset_index(drop=True)
    times = df["time_ns"].values
    xs = df["x_pix"].values.astype(np.int32)
    ys = df["y_pix"].values.astype(np.int32)

    chebyshev_thresh = 1 + max_gap

    # active clusters: maps (x, y) → cluster_id
    pixel_to_cluster: Dict[Tuple[int, int], int] = {}
    # cluster_id → ClusterState
    open_clusters: Dict[int, _ClusterState] = {}
    # union-find for merges
    uf = _UF()

    closed_clusters: List[_ClusterState] = []
    pixel_cluster_ids = np.full(len(df), -1, dtype=np.int64)

    next_id = 0
    close_time = -np.inf    # time at which we can close lagging clusters

    def _time_ok(t: float, state: _ClusterState) -> bool:
        if time_criterion == "chain":
            return (t - state.t_max) <= dt_ns
        else:   # "min"
            return (t - state.t_min) <= dt_ns

    def _close_old_clusters(current_time: float):
        nonlocal open_clusters, pixel_to_cluster
        to_close = []
        for cid, state in open_clusters.items():
            if current_time > state.t_max + dt_ns + safety_margin_ns:
                to_close.append(cid)
        for cid in to_close:
            state = open_clusters.pop(cid)
            closed_clusters.append(state)
            # remove pixel positions belonging to this cluster
            stale = [k for k, v in pixel_to_cluster.items() if uf.find(v) == uf.find(cid)]
            for k in stale:
                pixel_to_cluster.pop(k, None)

    for i in range(len(df)):
        t = float(times[i])
        x = int(xs[i])
        y = int(ys[i])

        _close_old_clusters(t)

        # find all neighbouring active clusters
        neighbour_ids = set()
        for dx in range(-chebyshev_thresh, chebyshev_thresh + 1):
            for dy in range(-chebyshev_thresh, chebyshev_thresh + 1):
                nb = pixel_to_cluster.get((x + dx, y + dy))
                if nb is not None:
                    root = uf.find(nb)
                    if root in open_clusters and _time_ok(t, open_clusters[root]):
                        neighbour_ids.add(root)

        if not neighbour_ids:
            # start new cluster
            cid = next_id
            next_id += 1
            state = _ClusterState(cid, t, i, time_criterion)
            open_clusters[cid] = state
            uf.find(cid)    # register in UF
            pixel_to_cluster[(x, y)] = cid
            pixel_cluster_ids[i] = cid
        else:
            # join the first cluster, merge the rest
            nb_list = list(neighbour_ids)
            target = nb_list[0]
            for other in nb_list[1:]:
                uf.union(target, other)
            root = uf.find(target)

            # update representative state
            if root not in open_clusters:
                # union made root point elsewhere; find the state
                for nb in nb_list:
                    if uf.find(nb) == root:
                        if nb in open_clusters:
                            open_clusters[root] = open_clusters.pop(nb)
                            break
                else:
                    # fallback: pick any
                    for nb in nb_list:
                        if nb in open_clusters:
                            open_clusters[root] = open_clusters.pop(nb)
                            break

            state = open_clusters[root]
            state.pixel_indices.append(i)
            state.t_min = min(state.t_min, t)
            state.t_max = max(state.t_max, t)
            if time_criterion == "chain":
                state.t_ref = state.t_max

            # remove old root entries for merged clusters
            for nb in nb_list:
                if nb != root and nb in open_clusters:
                    merged_state = open_clusters.pop(nb)
                    state.pixel_indices.extend(merged_state.pixel_indices)
                    state.t_min = min(state.t_min, merged_state.t_min)
                    state.t_max = max(state.t_max, merged_state.t_max)

            pixel_to_cluster[(x, y)] = root
            pixel_cluster_ids[i] = root

    # close remaining open clusters
    for cid, state in open_clusters.items():
        closed_clusters.append(state)

    # resolve pixel cluster IDs through UF
    for i in range(len(df)):
        pid = pixel_cluster_ids[i]
        if pid >= 0:
            pixel_cluster_ids[i] = uf.find(pid)

    # compact cluster IDs to 0, 1, 2, ...
    unique_ids = np.unique(pixel_cluster_ids[pixel_cluster_ids >= 0])
    id_map = {old: new for new, old in enumerate(unique_ids)}
    pixel_cluster_ids_compact = np.where(
        pixel_cluster_ids >= 0,
        np.vectorize(id_map.get)(pixel_cluster_ids, -1),
        -1,
    )

    df_out = df.copy()
    df_out["cluster_id"] = pixel_cluster_ids_compact

    # build cluster summary (minimal; features.py adds more)
    cluster_records = []
    for cid, new_id in id_map.items():
        mask = pixel_cluster_ids_compact == new_id
        sub = df_out[mask]
        cluster_records.append(
            {
                "cluster_id": new_id,
                "n_pixels": int(mask.sum()),
                "t_min": float(sub["time_ns"].min()),
                "t_max": float(sub["time_ns"].max()),
            }
        )

    clusters_df = pd.DataFrame(cluster_records).sort_values("cluster_id").reset_index(drop=True)
    log.info(
        "Clustering complete: %d pixels → %d clusters (Δt=%.0f ns, max_gap=%d)",
        len(df),
        len(clusters_df),
        dt_ns,
        max_gap,
    )
    return clusters_df, df_out


# ---------------------------------------------------------------------------
# Parameter-sensitivity sweep
# ---------------------------------------------------------------------------

def sweep(
    pixel_df: pd.DataFrame,
    dt_values: List[float],
    max_gap_values: List[int],
    plots_dir: Path,
) -> pd.DataFrame:
    """Sweep (Δt, max_gap) and return a summary DataFrame."""
    import matplotlib.pyplot as plt

    rows = []
    for dt in dt_values:
        for mg in max_gap_values:
            cl, _ = cluster_pixels(pixel_df, dt_ns=dt, max_gap=mg)
            sizes = cl["n_pixels"].values
            rows.append(
                {
                    "dt_ns": dt,
                    "max_gap": mg,
                    "n_clusters": len(cl),
                    "mean_size": float(sizes.mean()),
                    "median_size": float(np.median(sizes)),
                    "p95_size": float(np.percentile(sizes, 95)),
                }
            )

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mg in max_gap_values:
        sub = summary[summary["max_gap"] == mg]
        axes[0].plot(sub["dt_ns"], sub["n_clusters"], marker="o", label=f"gap={mg}")
        axes[1].plot(sub["dt_ns"], sub["mean_size"], marker="o", label=f"gap={mg}")
        axes[2].plot(sub["dt_ns"], sub["p95_size"], marker="o", label=f"gap={mg}")
    for ax, ylabel in zip(axes, ["# clusters", "mean size (px)", "p95 size (px)"]):
        ax.set_xlabel("Δt (ns)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(ylabel + " vs Δt")
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
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, load_pixels, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    pixel_df = load_pixels(cfg["paths"]["pixel_data"])
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    ccfg = cfg.get("clustering", {})
    clusters_df, cluster_pixels_df = cluster_pixels(
        pixel_df,
        dt_ns=ccfg.get("dt_ns", 2000),
        max_gap=ccfg.get("max_gap", 0),
        time_criterion=ccfg.get("time_criterion", "chain"),
        safety_margin_ns=ccfg.get("safety_margin_ns", 200),
    )

    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    cluster_pixels_df.to_parquet(out_dir / "cluster_pixels.parquet", index=False)
    print(f"Wrote {len(clusters_df)} clusters → {out_dir}/clusters.parquet")

    if args.sweep:
        sweep(
            pixel_df,
            dt_values=[200, 500, 1000, 2000],
            max_gap_values=[0, 1],
            plots_dir=plots_dir,
        )


if __name__ == "__main__":
    main()
