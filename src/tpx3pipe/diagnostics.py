"""Stage G — Contamination diagnostics.

Three analyses:
1. Baseline-under-peak estimate of mislabel fraction per class.
2. Within-bin morphology bimodality check (GMM + distribution plots).
3. Confident-disagreement map between classifier and ToF label.

Output: diagnostics_report.md with embedded figures.

Usage::

    python -m tpx3pipe.diagnostics --config config.yaml
"""

from __future__ import annotations

import argparse
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

log = logging.getLogger(__name__)

MORPHOLOGY_FEATURES = ["n_pixels", "tot_sum", "tot_max", "eccentricity"]


# ---------------------------------------------------------------------------
# 1. Baseline-under-peak mislabel estimate
# ---------------------------------------------------------------------------

def baseline_under_peak(
    clusters_df: pd.DataFrame,
    period_ns: float,
    pre_peak_window_ns: Tuple[float, float] = (0.0, 100.0),
) -> Dict[str, float]:
    """Estimate uniform-background mislabel fraction per class.

    Uses the pre-first-peak region as an estimate of the background density.
    """
    from typing import Tuple  # local to avoid top-level circular

    valid = clusters_df["folded_tof_ns"].notna()
    ftof = clusters_df.loc[valid, "folded_tof_ns"].values

    lo, hi = pre_peak_window_ns
    baseline_count = ((ftof >= lo) & (ftof < hi)).sum()
    window_width = hi - lo
    background_density = baseline_count / (len(ftof) * window_width)

    result: Dict[str, float] = {
        "background_density_per_ns": float(background_density),
        "baseline_window_ns": f"{lo}–{hi}",
    }

    # for each hard-label class, estimate contamination
    if "hard_label" not in clusters_df.columns:
        return result

    label_col = clusters_df.loc[valid, "hard_label"]
    ftof_s = pd.Series(ftof, index=clusters_df.index[valid])

    for lbl in sorted(label_col.dropna().unique()):
        if lbl < 0:
            continue
        mask = (label_col == lbl) & label_col.notna()
        bin_tofs = ftof_s[mask]
        if len(bin_tofs) == 0:
            continue
        bin_width_ns = bin_tofs.max() - bin_tofs.min()
        expected_bg = background_density * bin_width_ns * len(bin_tofs)
        contamination = min(1.0, expected_bg / max(len(bin_tofs), 1))
        result[f"class_{int(lbl)}_mislabel_frac"] = round(contamination, 4)
        result[f"class_{int(lbl)}_n_events"] = int(len(bin_tofs))

    return result


# ---------------------------------------------------------------------------
# 2. Within-bin morphology bimodality
# ---------------------------------------------------------------------------

def morphology_bimodality(
    clusters_df: pd.DataFrame,
    plots_dir: Path,
) -> Dict[str, dict]:
    """For each hard-label class, run 1–2 component GMM on morphology features."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "hard_label" not in clusters_df.columns:
        return {}

    results = {}
    avail_feats = [f for f in MORPHOLOGY_FEATURES if f in clusters_df.columns]
    valid = clusters_df["hard_label"].notna() & (clusters_df["hard_label"] >= 0)

    for lbl in sorted(clusters_df.loc[valid, "hard_label"].unique()):
        mask = valid & (clusters_df["hard_label"] == lbl)
        sub = clusters_df[mask][avail_feats].dropna()
        if len(sub) < 10:
            continue

        bimodal_flags = {}
        fig, axes = plt.subplots(1, len(avail_feats), figsize=(4 * len(avail_feats), 4))
        if len(avail_feats) == 1:
            axes = [axes]

        for ax, feat in zip(axes, avail_feats):
            X = sub[feat].values.reshape(-1, 1)
            X_log = np.log1p(X - X.min() + 1e-3)

            bic1 = GaussianMixture(n_components=1, random_state=0).fit(X_log).bic(X_log)
            bic2 = GaussianMixture(n_components=2, random_state=0).fit(X_log).bic(X_log)
            is_bimodal = bic2 < bic1 - 2   # BIC improvement threshold

            gm2 = GaussianMixture(n_components=2, random_state=0).fit(X_log)
            minority_frac = min(gm2.weights_)

            bimodal_flags[feat] = {
                "bimodal": bool(is_bimodal),
                "bic_1comp": float(bic1),
                "bic_2comp": float(bic2),
                "minority_frac": float(minority_frac) if is_bimodal else 0.0,
            }

            ax.hist(sub[feat], bins=50, color="steelblue", alpha=0.7)
            title = f"{feat}\n{'BIMODAL' if is_bimodal else 'unimodal'}"
            if is_bimodal:
                title += f" (min={minority_frac:.2f})"
            ax.set_title(title)
            ax.set_xlabel(feat)

        fig.suptitle(f"Class {int(lbl)} morphology distributions")
        fig.tight_layout()
        fig.savefig(plots_dir / f"morphology_class_{int(lbl)}.png", dpi=100)
        plt.close(fig)

        results[f"class_{int(lbl)}"] = bimodal_flags

    return results


# ---------------------------------------------------------------------------
# 3. Confident-disagreement map
# ---------------------------------------------------------------------------

def confident_disagreement(
    clusters_df: pd.DataFrame,
    pred_col: str = "predicted_label",
    confidence_thresh: float = 0.80,
) -> Optional[pd.DataFrame]:
    """Cross-tab: events where classifier confidently disagrees with ToF label."""
    needed = {"hard_label", pred_col, "label_confidence"}
    if not needed.issubset(clusters_df.columns):
        log.warning("confident_disagreement: missing columns %s", needed - set(clusters_df.columns))
        return None

    valid = (
        clusters_df["hard_label"].notna()
        & (clusters_df["hard_label"] >= 0)
        & clusters_df[pred_col].notna()
        & (clusters_df["label_confidence"] >= confidence_thresh)
    )
    sub = clusters_df[valid].copy()
    sub["agree"] = sub["hard_label"] == sub[pred_col]
    disagreements = sub[~sub["agree"]]

    if len(disagreements) == 0:
        return None

    ct = pd.crosstab(
        disagreements["hard_label"].astype(int),
        disagreements[pred_col].astype(int),
        rownames=["tof_label"],
        colnames=["predicted_label"],
    )
    return ct


# ---------------------------------------------------------------------------
# Validation against ground truth
# ---------------------------------------------------------------------------

def validate_against_truth(
    clusters_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    estimated_contamination: Dict,
) -> Dict:
    """Compare estimated mislabel fraction to ground-truth wrap_count."""
    if "cluster_id" not in truth_df.columns or "wrap_count" not in truth_df.columns:
        return {}

    merged = clusters_df.merge(truth_df[["cluster_id", "wrap_count"]], on="cluster_id", how="inner")
    if len(merged) == 0:
        return {}

    true_contamination = (merged["wrap_count"] > 0).mean()
    estimated = estimated_contamination.get("class_0_mislabel_frac", np.nan)

    return {
        "true_contamination_frac": float(true_contamination),
        "estimated_contamination_frac": float(estimated),
        "error": float(abs(true_contamination - estimated)) if not np.isnan(estimated) else np.nan,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fig_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_report(
    baseline_result: dict,
    bimodality_result: dict,
    disagreement_ct,
    validation_result: dict,
    plots_dir: Path,
    out_path: Path,
) -> None:
    lines = [
        "# Contamination Diagnostics Report",
        "",
        "> **Language note:** labels refer to folded-ToF bins, not particle species.",
        "> Species interpretation requires SME input (see Open Questions).",
        "",
        "## 1. Baseline-Under-Peak Mislabel Estimate",
        "",
        f"Background density: {baseline_result.get('background_density_per_ns', 'N/A'):.4e} events/ns",
        f"Baseline window: {baseline_result.get('baseline_window_ns', 'N/A')}",
        "",
    ]

    for key, val in baseline_result.items():
        if key.startswith("class_") and "mislabel_frac" in key:
            cls = key.split("_")[1]
            n = baseline_result.get(f"class_{cls}_n_events", "?")
            lines.append(f"- **Class {cls}**: estimated mislabel fraction = {val:.4f}  (n={n})")
    lines.append("")

    if validation_result:
        lines += [
            "### Validation against synthetic ground truth",
            "",
            f"- True contamination fraction: {validation_result.get('true_contamination_frac', 'N/A'):.4f}",
            f"- Estimated contamination fraction: {validation_result.get('estimated_contamination_frac', 'N/A'):.4f}",
            f"- Absolute error: {validation_result.get('error', 'N/A'):.4f}",
            "",
        ]

    lines += ["## 2. Within-Bin Morphology Bimodality", ""]
    for cls, feats in bimodality_result.items():
        lines.append(f"### {cls}")
        for feat, info in feats.items():
            flag = "**BIMODAL**" if info["bimodal"] else "unimodal"
            lines.append(
                f"- {feat}: {flag}"
                + (f", minority fraction ≈ {info['minority_frac']:.3f}" if info["bimodal"] else "")
            )
        img_path = plots_dir / f"morphology_{cls}.png"
        if img_path.exists():
            lines.append(f"\n![{cls} morphology]({img_path})\n")
    lines.append("")

    lines += ["## 3. Confident-Disagreement Map", ""]
    if disagreement_ct is not None:
        lines.append("Cross-tab of classifier predictions vs ToF labels (confident events only):\n")
        lines.append("```")
        lines.append(disagreement_ct.to_string())
        lines.append("```\n")
    else:
        lines.append("*Not available (requires trained classifier output in clusters.parquet).*\n")

    lines += [
        "## Open Questions for SMEs",
        "",
        "- **Flight path and trigger offset**: unknown — prevents ToF-to-species conversion.",
        "- **Source pulse structure and expected species**: not provided.",
        "- **Energy calibration**: no per-pixel calibration available; raw ToT used.",
        "- **Time-walk correction**: no correction applied; see `dt_min_vs_argmax` diagnostic.",
        "- **Expected maximum ToF**: determines whether a 1× period fold is sufficient.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Diagnostics report written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# import Tuple here to avoid NameError at module-level function
from typing import Tuple


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage G: contamination diagnostics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--truth", default=None, help="Path to truth.parquet (optional)")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    from tpx3pipe.tdc import load_period
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")
    period_ns = load_period(out_dir, cfg)

    baseline = baseline_under_peak(clusters_df, period_ns)
    bimodality = morphology_bimodality(clusters_df, plots_dir)
    disagreement = confident_disagreement(clusters_df)

    validation = {}
    truth_path = args.truth or (Path("data") / "truth.parquet")
    if Path(truth_path).exists():
        truth_df = pd.read_parquet(truth_path)
        validation = validate_against_truth(clusters_df, truth_df, baseline)

    generate_report(
        baseline, bimodality, disagreement, validation,
        plots_dir, out_dir / "diagnostics_report.md",
    )
    print(f"Diagnostics report: {out_dir}/diagnostics_report.md")


if __name__ == "__main__":
    main()
