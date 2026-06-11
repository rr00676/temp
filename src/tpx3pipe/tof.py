"""Stage D — Time-of-Flight computation and folding.

- Raw ToF: cluster_time − most recent TDC edge ≤ cluster_time (searchsorted).
- Folded ToF: raw_tof mod measured_period.
- Reports how often raw_tof and folded_tof differ beyond a rounding tolerance
  (proxy for missed-edge impact).
- Produces: folded-ToF spectrum plot, ToF × ToT bivariate histogram.

Usage::

    python -m tpx3pipe.tof --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_tof(
    clusters_df: pd.DataFrame,
    tdc_df: pd.DataFrame,
    period_ns: float,
    cluster_time_mode: str = "t_min",
) -> pd.DataFrame:
    """Append raw_tof_ns and folded_tof_ns columns to clusters_df.

    Parameters
    ----------
    clusters_df : must contain t_min and t_argmax_tot columns
    tdc_df      : must contain time_ns column (sorted)
    period_ns   : measured beam period
    cluster_time_mode : "t_min" | "t_argmax_tot"
    """
    tdc_times = np.sort(tdc_df["time_ns"].values)

    if cluster_time_mode == "t_min":
        cluster_times = clusters_df["t_min"].values
    elif cluster_time_mode == "t_argmax_tot":
        if "t_argmax_tot" not in clusters_df.columns:
            raise ValueError("t_argmax_tot not in clusters_df — run Stage C first.")
        cluster_times = clusters_df["t_argmax_tot"].values
    else:
        raise ValueError(f"Unknown cluster_time_mode: {cluster_time_mode!r}")

    # searchsorted: find index of most-recent TDC edge ≤ cluster_time
    # side='right' gives insertion point after equal elements → subtract 1
    idx = np.searchsorted(tdc_times, cluster_times, side="right") - 1
    # clusters before first TDC edge get idx = -1 → assign NaN
    valid = idx >= 0
    raw_tof = np.full(len(cluster_times), np.nan)
    raw_tof[valid] = cluster_times[valid] - tdc_times[idx[valid]]

    folded_tof = raw_tof % period_ns

    # diagnostic: how often do raw ≠ folded (missed-edge impact)
    diff = np.abs(raw_tof - folded_tof)
    n_differ = int((diff > 0.5).sum())   # > 0.5 ns tolerance

    out = clusters_df.copy()
    out["cluster_time_ns"] = cluster_times
    out["raw_tof_ns"] = raw_tof
    out["folded_tof_ns"] = folded_tof
    out["_tof_valid"] = valid

    log.info(
        "ToF computed: %d clusters, %.1f%% before first TDC edge, "
        "raw≠folded for %d clusters (%.2f%%)",
        len(out),
        100 * (~valid).mean(),
        n_differ,
        100 * n_differ / max(valid.sum(), 1),
    )
    return out


def plot_tof_diagnostics(
    clusters_df: pd.DataFrame,
    period_ns: float,
    plots_dir: Path,
) -> None:
    """Folded-ToF spectrum and ToF × ToT bivariate histogram."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    valid = clusters_df["folded_tof_ns"].notna()
    ftof = clusters_df.loc[valid, "folded_tof_ns"].values
    tot_sum = clusters_df.loc[valid, "tot_sum"].values if "tot_sum" in clusters_df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(0, period_ns, 200)
    ax.hist(ftof, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xlabel("Folded ToF (ns)")
    ax.set_ylabel("Clusters / bin")
    ax.set_title("Folded ToF spectrum")

    ax = axes[1]
    if tot_sum is not None:
        h, xedges, yedges = np.histogram2d(
            ftof,
            np.log1p(tot_sum),
            bins=[np.linspace(0, period_ns, 150), np.linspace(0, np.log1p(tot_sum.max()), 100)],
        )
        ax.imshow(
            h.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="inferno",
        )
        ax.set_xlabel("Folded ToF (ns)")
        ax.set_ylabel("log1p(tot_sum)")
        ax.set_title("ToF × ToT bivariate histogram")
    else:
        ax.text(0.5, 0.5, "tot_sum not available", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(plots_dir / "tof_spectrum.png", dpi=120)
    plt.close(fig)
    log.info("ToF diagnostic plots saved")

    # secondary: compare t_min vs t_argmax_tot based ToF if both present
    if "t_argmax_tot" in clusters_df.columns and "t_min" in clusters_df.columns:
        tdc_placeholder = None   # skip if we'd need to recompute; already done in main


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage D: compute ToF and fold")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, load_tdc, ensure_artifacts_dir, ensure_plots_dir
    from tpx3pipe.tdc import load_period
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    tdc_df = load_tdc(cfg["paths"]["tdc_data"])
    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")
    period_ns = load_period(out_dir, cfg)

    mode = cfg.get("tof", {}).get("cluster_time_mode", "t_min")
    clusters_df = compute_tof(clusters_df, tdc_df, period_ns, cluster_time_mode=mode)

    # also compute with the other mode for diagnostic comparison
    alt_mode = "t_argmax_tot" if mode == "t_min" else "t_min"
    if alt_mode in clusters_df.columns:
        alt = compute_tof(clusters_df, tdc_df, period_ns, cluster_time_mode=alt_mode)
        diff = (clusters_df["folded_tof_ns"] - alt["folded_tof_ns"]).abs()
        log.info(
            "ToF mode comparison (%s vs %s): mean diff=%.2f ns, max diff=%.2f ns",
            mode, alt_mode, diff.mean(), diff.max(),
        )

    plot_tof_diagnostics(clusters_df, period_ns, plots_dir)
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    print(f"Updated {out_dir}/clusters.parquet with ToF columns")


if __name__ == "__main__":
    main()
