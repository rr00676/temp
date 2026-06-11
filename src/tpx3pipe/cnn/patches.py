"""Patch extraction for CNN input (Stage I).

Each cluster is represented as a P×P×2 image:
  ch0 = log1p(ToT)
  ch1 = time_ns - cluster_t_min  (clipped at Δt_ns)

Centered on the max-ToT pixel.

Leakage rules enforced here:
- No TDC-derived quantity (no raw/folded ToF).
- No absolute time (ch1 is RELATIVE to cluster start).
- No chip (x, y) position in the tensor (patch is centered/cropped).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def extract_patches(
    cluster_pixels_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    patch_size: int = 32,
    dt_clip_ns: float = 2000.0,
    channel_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Extract P×P×2 patches for all clusters.

    Returns
    -------
    patches : (N, 2, P, P) float32 array (channel-first for PyTorch)
    cluster_ids : (N,) int array matching row order
    channel_stats : (mean_per_channel, std_per_channel) arrays of shape (2,)
                    Pass these back in on val/test to use train-set normalization.
    """
    P = patch_size
    half = P // 2

    required_pix = {"cluster_id", "x_pix", "y_pix", "time_ns", "tot"}
    required_cl = {"cluster_id", "t_min"}
    missing = (required_pix - set(cluster_pixels_df.columns)) | (required_cl - set(clusters_df.columns))
    if missing:
        raise ValueError(f"Missing columns for patch extraction: {missing}")

    cl_lookup = clusters_df.set_index("cluster_id")[["t_min"]].to_dict("index")

    patches_list = []
    ids_list = []

    grouped = cluster_pixels_df.groupby("cluster_id", sort=True)
    for cid, grp in grouped:
        if cid not in cl_lookup:
            continue
        t_min = cl_lookup[cid]["t_min"]

        x = grp["x_pix"].values.astype(np.int32)
        y = grp["y_pix"].values.astype(np.int32)
        tot = grp["tot"].values.astype(np.float32)
        t = grp["time_ns"].values.astype(np.float32)

        # center on max-ToT pixel
        center_idx = np.argmax(tot)
        cx, cy = int(x[center_idx]), int(y[center_idx])

        ch0 = np.zeros((P, P), dtype=np.float32)   # log1p(ToT)
        ch1 = np.zeros((P, P), dtype=np.float32)   # relative time, clipped

        for xi, yi, ti, ti_abs in zip(x, y, tot, t):
            px = (xi - cx) + half
            py = (yi - cy) + half
            if 0 <= px < P and 0 <= py < P:
                ch0[py, px] = np.log1p(ti)
                ch1[py, px] = float(np.clip(ti_abs - t_min, 0.0, dt_clip_ns))

        patch = np.stack([ch0, ch1], axis=0)   # (2, P, P)
        patches_list.append(patch)
        ids_list.append(cid)

    if not patches_list:
        raise RuntimeError("No patches extracted — check cluster_pixels_df content.")

    patches = np.stack(patches_list, axis=0)   # (N, 2, P, P)
    cluster_ids = np.array(ids_list, dtype=np.int64)

    if channel_stats is None:
        mean = patches.mean(axis=(0, 2, 3))   # (2,)
        std = patches.std(axis=(0, 2, 3)) + 1e-6
        channel_stats = (mean, std)
    else:
        mean, std = channel_stats

    patches = (patches - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]

    coverage = (patches_list.__len__() / max(len(clusters_df), 1)) * 100
    log.info("Extracted %d patches (P=%d), coverage=%.1f%%", len(patches_list), P, coverage)
    return patches, cluster_ids, channel_stats


def dihedral_augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random dihedral transform (8 elements) to a (C, H, W) patch."""
    k = int(rng.integers(0, 4))           # 0–3 rotations of 90°
    patch = np.rot90(patch, k=k, axes=(1, 2))
    if rng.random() > 0.5:
        patch = np.flip(patch, axis=2).copy()   # horizontal flip
    return patch
