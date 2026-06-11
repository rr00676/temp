"""Synthetic data generator (Stage E).

Produces (pixel_df, tdc_df) matching the real Timepix3 schema, plus a ground-truth
dataframe that is NEVER consumed by pipeline stages — only by tests and diagnostics
validation.

Usage::

    python -m tpx3pipe.synth --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TDC generation
# ---------------------------------------------------------------------------

def _make_tdc(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    sc = cfg["synth"]
    period = sc["period_ns"]
    jitter = sc["period_jitter_ns"]
    duration_ns = sc["duration_s"] * 1e9
    missed_rate = sc.get("missed_edge_rate", 0.0)

    n_pulses = int(duration_ns / period) + 1
    # actual inter-pulse gaps (with jitter)
    gaps = period + rng.normal(0, jitter, size=n_pulses)
    gaps = np.maximum(gaps, period * 0.5)
    times = np.cumsum(np.concatenate([[0.0], gaps]))

    # randomly drop some edges to simulate missed-edge pathology
    if missed_rate > 0:
        keep = rng.random(len(times)) >= missed_rate
        times = times[keep]

    n = len(times)
    tdc_df = pd.DataFrame({
        "tdc_type":   np.zeros(n, dtype=np.int32),
        "timestamp":  np.zeros(n, dtype=np.int64),
        "fine_stamp": np.zeros(n, dtype=np.int32),
        "counter":    np.arange(n, dtype=np.int64),
        "time_ns":    times,
    })
    return tdc_df


# ---------------------------------------------------------------------------
# Per-archetype cluster pixel layout
# ---------------------------------------------------------------------------

def _dot_pixels(n_pix: int, rng: np.random.Generator) -> np.ndarray:
    """Returns (n_pix, 2) array of (x, y) offsets, compact dot shape."""
    coords = {(0, 0)}
    while len(coords) < n_pix:
        x, y = rng.integers(-1, 2, size=2)
        coords.add((int(x), int(y)))
    return np.array(list(coords)[:n_pix])


def _curl_pixels(n_pix: int, rng: np.random.Generator) -> np.ndarray:
    """Random-walk track."""
    coords = [(0, 0)]
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for _ in range(n_pix - 1):
        x, y = coords[-1]
        dx, dy = directions[rng.integers(0, len(directions))]
        coords.append((x + dx, y + dy))
    return np.array(coords)


def _blob_pixels(n_pix: int, rng: np.random.Generator) -> np.ndarray:
    """Compact round cluster grown outward from origin."""
    coords = {(0, 0)}
    radius = 1
    while len(coords) < n_pix:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    coords.add((dx, dy))
        radius += 1
    arr = np.array(list(coords))
    # randomly select n_pix if we have more
    if len(arr) > n_pix:
        idx = rng.choice(len(arr), size=n_pix, replace=False)
        arr = arr[idx]
    return arr


_ARCHETYPE_FUNS = {"dot": _dot_pixels, "curl": _curl_pixels, "blob": _blob_pixels}


def _make_cluster_pixels(
    archetype: str,
    archetype_cfg: dict,
    cx: int,
    cy: int,
    cluster_time_ns: float,
    intra_spread: float,
    tw_delay: float,
    tw_thresh: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate pixel rows for a single cluster."""
    ac = archetype_cfg[archetype]
    lo, hi = ac["n_pixels_range"]
    n_pix = int(rng.integers(lo, hi + 1))

    offsets = _ARCHETYPE_FUNS[archetype](n_pix, rng)

    # pixel positions (clamp to 0–255)
    xs = np.clip(cx + offsets[:, 0], 0, 255).astype(np.int32)
    ys = np.clip(cy + offsets[:, 1], 0, 255).astype(np.int32)

    # ToT: blob has radial falloff, others flat-ish
    tot_mean = ac["tot_mean"]
    tot_std = ac["tot_std"]
    if archetype == "blob":
        r2 = offsets[:, 0] ** 2 + offsets[:, 1] ** 2
        r2_max = max(r2.max(), 1)
        scale = 1.0 - 0.6 * r2 / r2_max
        tots = np.maximum(1, (tot_mean * scale + rng.normal(0, tot_std, n_pix)).astype(np.int32))
    else:
        tots = np.maximum(1, rng.normal(tot_mean, tot_std, n_pix).astype(np.int32))

    # pixel times: cluster_time + small intra-cluster spread
    pixel_times = cluster_time_ns + rng.exponential(intra_spread / n_pix, size=n_pix)
    # time-walk: low-ToT pixels arrive later
    tw_mask = tots < tw_thresh
    pixel_times[tw_mask] += tw_delay * (1.0 - tots[tw_mask] / tw_thresh)

    df = pd.DataFrame(
        {
            "x_pix":      xs,
            "y_pix":      ys,
            "toa":        np.zeros(n_pix, dtype=np.int64),   # opaque — not used
            "ftoa":       np.zeros(n_pix, dtype=np.int64),
            "spidr_time": np.zeros(n_pix, dtype=np.int64),
            "time_ns":    pixel_times,
            "tot":        tots,   # required by features/CNN; present in real Timepix3 data
        }
    )
    return df


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (pixel_df, tdc_df, truth_df).

    truth_df columns: cluster_id, species, true_tof_ns, wrap_count, pulse_time_ns
    truth_df is ground truth and must never be fed to pipeline stages.
    """
    if seed is None:
        seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    sc = cfg["synth"]
    period = sc["period_ns"]
    intra_spread = sc.get("intra_cluster_spread_ns", 50.0)
    tw_delay = sc.get("time_walk_delay_ns", 80.0)
    tw_thresh = sc.get("time_walk_threshold_tot", 100)
    shuffle_horizon = sc.get("shuffle_horizon_ns", 200_000.0)
    archetype_cfg = sc["archetypes"]

    tdc_df = _make_tdc(cfg, rng)
    pulse_times = tdc_df["time_ns"].values
    duration_ns = sc["duration_s"] * 1e9

    all_pixels: List[pd.DataFrame] = []
    truth_rows: List[dict] = []
    cluster_id = 0
    n_pulses = len(pulse_times)

    for species_cfg in sc["species"]:
        name = species_cfg["name"]
        rate = species_cfg["rate_per_pulse"]
        tof_mean = species_cfg["tof_mean_ns"]
        tof_std = species_cfg["tof_std_ns"]
        archetype = species_cfg["archetype"]

        # --- vectorized: draw total cluster count once, then sample pulses ---
        # Total expected clusters = rate * n_pulses; draw from Poisson
        n_total = int(rng.poisson(rate * n_pulses))
        if n_total == 0:
            continue

        # sample a pulse for each cluster (uniform over pulse_times)
        pulse_indices = rng.integers(0, n_pulses, size=n_total)
        pulse_t_arr = pulse_times[pulse_indices]

        # draw ToF values
        true_tof_arr = np.maximum(0.0, rng.normal(tof_mean, tof_std, size=n_total))
        cluster_t_arr = pulse_t_arr + true_tof_arr

        # filter clusters that fall outside the acquisition window
        keep = cluster_t_arr <= duration_ns
        pulse_t_arr = pulse_t_arr[keep]
        true_tof_arr = true_tof_arr[keep]
        cluster_t_arr = cluster_t_arr[keep]
        n_kept = int(keep.sum())

        # random centroids for all clusters at once
        cxs = rng.integers(5, 251, size=n_kept)
        cys = rng.integers(5, 251, size=n_kept)

        for i in range(n_kept):
            wrap_count = int(true_tof_arr[i] / period)
            pix_df = _make_cluster_pixels(
                archetype, archetype_cfg, int(cxs[i]), int(cys[i]),
                float(cluster_t_arr[i]), intra_spread, tw_delay, tw_thresh, rng,
            )
            pix_df["_cluster_id"] = cluster_id
            all_pixels.append(pix_df)
            truth_rows.append(
                {
                    "cluster_id": cluster_id,
                    "species": name,
                    "true_tof_ns": float(true_tof_arr[i]),
                    "wrap_count": wrap_count,
                    "pulse_time_ns": float(pulse_t_arr[i]),
                }
            )
            cluster_id += 1

    if not all_pixels:
        raise RuntimeError("No clusters generated — check synth config (rates, duration).")

    pixel_df = pd.concat(all_pixels, ignore_index=True)

    # optionally shuffle within a 200 µs horizon to emulate unsorted arrival
    if shuffle_horizon > 0:
        bucket = (pixel_df["time_ns"].values // shuffle_horizon).astype(np.int64)
        order = np.arange(len(pixel_df))
        for b in np.unique(bucket):
            idx = np.where(bucket == b)[0]
            shuffled_idx = idx.copy()
            rng.shuffle(shuffled_idx)
            order[idx] = order[shuffled_idx]
        pixel_df = pixel_df.iloc[order].reset_index(drop=True)

    # drop internal column
    pixel_df = pixel_df.drop(columns=["_cluster_id"]).reset_index(drop=True)

    truth_df = pd.DataFrame(truth_rows)
    log.info(
        "Generated %d pixel hits, %d TDC edges, %d clusters from %d species",
        len(pixel_df),
        len(tdc_df),
        len(truth_df),
        len(sc["species"]),
    )
    return pixel_df, tdc_df, truth_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic Timepix3 data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config
    cfg = load_config(args.config)

    pixel_df, tdc_df, truth_df = generate(cfg, seed=args.seed)

    print(f"pixel_df  : {len(pixel_df):,} rows  columns={list(pixel_df.columns)}")
    print(f"tdc_df    : {len(tdc_df):,} rows  columns={list(tdc_df.columns)}")
    print(f"truth_df  : {len(truth_df):,} clusters")
    return pixel_df, tdc_df, truth_df


if __name__ == "__main__":
    main()
