"""Stage C — Cluster feature extraction.

Computes per-cluster scalar features from the pixel-level data.  The resulting
features are appended to clusters.parquet and used by Stages H (GBT) and I (CNN).

Features computed
-----------------
n_pixels, tot_sum, tot_max, tot_mean,
cx_w, cy_w          (ToT-weighted centroid),
bbox_w, bbox_h      (bounding-box width/height),
eccentricity        (PCA-based; 0=round, 1=linear),
linearity           (ratio of first PC eigenvalue to total variance),
compactness         (n_pixels / (bbox_w × bbox_h); proxy for pixel density),
t_min               (earliest pixel time_ns in cluster),
t_argmax_tot        (time_ns of max-ToT pixel),
dt_min_vs_argmax    (t_argmax_tot − t_min; time-walk indicator)

Usage::

    python -m tpx3pipe.features --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_features(cluster_pixels_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster features.

    Parameters
    ----------
    cluster_pixels_df : must contain columns
        cluster_id, x_pix, y_pix, time_ns, tot

    Returns
    -------
    DataFrame indexed by cluster_id with all feature columns.
    """
    required = {"cluster_id", "x_pix", "y_pix", "time_ns", "tot"}
    missing = required - set(cluster_pixels_df.columns)
    if missing:
        raise ValueError(f"cluster_pixels_df missing columns: {missing}")

    records = []
    grouped = cluster_pixels_df.groupby("cluster_id", sort=True)

    for cid, grp in grouped:
        x = grp["x_pix"].values.astype(float)
        y = grp["y_pix"].values.astype(float)
        t = grp["time_ns"].values
        tot = grp["tot"].values.astype(float)
        n = len(grp)

        tot_sum = tot.sum()
        tot_max = tot.max()
        tot_mean = tot.mean()

        # ToT-weighted centroid
        w = tot / max(tot_sum, 1e-9)
        cx_w = (x * w).sum()
        cy_w = (y * w).sum()

        # bounding box
        bbox_w = float(x.max() - x.min() + 1)
        bbox_h = float(y.max() - y.min() + 1)
        compactness = n / (bbox_w * bbox_h)

        # PCA (ToT-weighted)
        if n >= 2:
            xc = x - cx_w
            yc = y - cy_w
            cov = np.array([
                [(w * xc * xc).sum(), (w * xc * yc).sum()],
                [(w * xc * yc).sum(), (w * yc * yc).sum()],
            ])
            eigvals = np.linalg.eigvalsh(cov)   # sorted ascending
            l1, l2 = eigvals[0], eigvals[1]
            total_var = l1 + l2
            if total_var > 1e-12:
                linearity = l2 / total_var
                eccentricity = np.sqrt(max(0.0, 1.0 - l1 / max(l2, 1e-12)))
            else:
                linearity = 0.0
                eccentricity = 0.0
        else:
            linearity = 0.0
            eccentricity = 0.0

        # timing
        t_min = float(t.min())
        idx_max_tot = np.argmax(tot)
        t_argmax_tot = float(t[idx_max_tot])
        dt_min_vs_argmax = t_argmax_tot - t_min

        records.append(
            {
                "cluster_id": int(cid),
                "n_pixels": n,
                "tot_sum": float(tot_sum),
                "tot_max": float(tot_max),
                "tot_mean": float(tot_mean),
                "cx_w": float(cx_w),
                "cy_w": float(cy_w),
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "eccentricity": float(eccentricity),
                "linearity": float(linearity),
                "compactness": float(compactness),
                "t_min": t_min,
                "t_argmax_tot": t_argmax_tot,
                "dt_min_vs_argmax": float(dt_min_vs_argmax),
            }
        )

    feat_df = pd.DataFrame(records)
    log.info("Computed features for %d clusters", len(feat_df))
    return feat_df


SCALAR_FEATURE_COLS = [
    "n_pixels", "tot_sum", "tot_max", "tot_mean",
    "cx_w", "cy_w", "bbox_w", "bbox_h",
    "eccentricity", "linearity", "compactness",
    "dt_min_vs_argmax",
]

CNN_SCALAR_COLS = ["n_pixels", "tot_sum", "tot_max", "eccentricity"]


def plot_feature_diagnostics(feat_df: pd.DataFrame, plots_dir: Path) -> None:
    """Plot dt_min_vs_argmax distribution overall and vs n_pixels."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(feat_df["dt_min_vs_argmax"].clip(-200, 2000), bins=100, color="steelblue")
    ax.set_xlabel("dt_min_vs_argmax (ns)")
    ax.set_ylabel("Count")
    ax.set_title("Time-walk indicator distribution")

    ax = axes[1]
    sc = ax.scatter(
        feat_df["n_pixels"],
        feat_df["dt_min_vs_argmax"].clip(-200, 2000),
        s=4,
        alpha=0.3,
        c=feat_df["tot_mean"],
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax, label="tot_mean")
    ax.set_xlabel("n_pixels")
    ax.set_ylabel("dt_min_vs_argmax (ns)")
    ax.set_title("Time-walk vs cluster size")

    fig.tight_layout()
    fig.savefig(plots_dir / "feature_timewalk.png", dpi=120)
    plt.close(fig)
    log.info("Feature diagnostic plot saved")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage C: compute cluster features")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    cluster_pixels_df = pd.read_parquet(out_dir / "cluster_pixels.parquet")
    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")

    feat_df = compute_features(cluster_pixels_df)
    plot_feature_diagnostics(feat_df, plots_dir)

    # merge features into clusters (update clusters.parquet)
    # drop any existing feature columns that overlap to avoid duplicates
    overlap = set(clusters_df.columns) & set(feat_df.columns) - {"cluster_id"}
    clusters_df = clusters_df.drop(columns=list(overlap))
    clusters_df = clusters_df.merge(feat_df, on="cluster_id", how="left")
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    print(f"Updated {out_dir}/clusters.parquet with {len(feat_df.columns)-1} feature columns")


if __name__ == "__main__":
    main()
