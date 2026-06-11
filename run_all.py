"""End-to-end pipeline runner.

Executes all stages in order on synthetic data (or real data if paths are set in
config.yaml).  Equivalent to ``make all``.

Usage::

    python run_all.py [--config config.yaml] [--skip-cnn] [--synth-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")


def _step(name: str, fn, *args, **kwargs):
    log.info("=== %s START ===", name)
    t0 = time.time()
    result = fn(*args, **kwargs)
    log.info("=== %s DONE in %.1f s ===", name, time.time() - t0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run full tpx3pipe pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training (Stage I)")
    parser.add_argument("--synth-only", action="store_true",
                        help="Only generate synthetic data, then stop")
    args = parser.parse_args()

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    # -------------------------------------------------------------------------
    # Data loading: real TPX3 file, existing parquet/csv, or synth generator
    # -------------------------------------------------------------------------
    import pandas as pd

    tpx3_cfg = cfg.get("tpx3", {})
    tpx3_file = tpx3_cfg.get("file")
    truth_df = pd.DataFrame()   # empty unless synth

    if tpx3_file:
        # --- Real data via readtpx3.py ---
        from tpx3pipe.io import load_from_tpx3
        pixel_df, tdc_df = _step(
            "Load TPX3 file", load_from_tpx3,
            tpx3_file,
            module_path=tpx3_cfg.get("module_path"),
            column_map=tpx3_cfg.get("column_map") or {},
        )
        print(f"  {len(pixel_df)} pixels, {len(tdc_df)} TDC edges (real data)")

        if args.synth_only:
            print("--synth-only has no effect when tpx3.file is set.")
            return

    elif not args.synth_only and Path(cfg["paths"]["pixel_data"]).exists():
        # --- Pre-existing parquet/csv files ---
        from tpx3pipe.io import load_pixels, load_tdc
        pixel_df = _step("Load pixels", load_pixels, cfg["paths"]["pixel_data"])
        tdc_df   = _step("Load TDC",    load_tdc,    cfg["paths"]["tdc_data"])
        print(f"  {len(pixel_df)} pixels, {len(tdc_df)} TDC edges (from files)")

    else:
        # --- Synthetic data generator (Stage E) ---
        from tpx3pipe.synth import generate
        pixel_df, tdc_df, truth_df = _step("Stage E (synth)", generate, cfg)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        pixel_df.to_parquet(data_dir / "pixels.parquet", index=False)
        tdc_df.to_parquet(data_dir / "tdc.parquet", index=False)
        truth_df.to_parquet(data_dir / "truth.parquet", index=False)
        print(f"  {len(pixel_df)} pixels, {len(tdc_df)} TDC edges, {len(truth_df)} truth clusters (synth)")

        if args.synth_only:
            return

    # -------------------------------------------------------------------------
    # Stage A: TDC diagnostics
    # -------------------------------------------------------------------------
    from tpx3pipe.tdc import run_tdc_diagnostics
    tdc_report = _step("Stage A (TDC)", run_tdc_diagnostics, tdc_df, cfg, out_dir, plots_dir)
    period_ns = tdc_report["measured_period_ns"]

    # -------------------------------------------------------------------------
    # Stage B: clustering
    # -------------------------------------------------------------------------
    from tpx3pipe.clustering import cluster_pixels
    ccfg = cfg.get("clustering", {})
    clusters_df, cluster_pixels_df = _step(
        "Stage B (cluster)", cluster_pixels, pixel_df,
        dt_ns=ccfg.get("dt_ns", 2000),
        max_gap=ccfg.get("max_gap", 0),
        time_criterion=ccfg.get("time_criterion", "chain"),
        safety_margin_ns=ccfg.get("safety_margin_ns", 200),
    )
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    cluster_pixels_df.to_parquet(out_dir / "cluster_pixels.parquet", index=False)
    print(f"  {len(clusters_df)} clusters")

    # -------------------------------------------------------------------------
    # Stage C: features
    # -------------------------------------------------------------------------
    from tpx3pipe.features import compute_features, plot_feature_diagnostics
    feat_df = _step("Stage C (features)", compute_features, cluster_pixels_df)
    _step("Stage C (plots)", plot_feature_diagnostics, feat_df, plots_dir)
    overlap = set(clusters_df.columns) & set(feat_df.columns) - {"cluster_id"}
    clusters_df = clusters_df.drop(columns=list(overlap))
    clusters_df = clusters_df.merge(feat_df, on="cluster_id", how="left")

    # -------------------------------------------------------------------------
    # Stage D: ToF
    # -------------------------------------------------------------------------
    from tpx3pipe.tof import compute_tof, plot_tof_diagnostics
    mode = cfg.get("tof", {}).get("cluster_time_mode", "t_min")
    clusters_df = _step("Stage D (ToF)", compute_tof, clusters_df, tdc_df, period_ns, mode)
    _step("Stage D (plots)", plot_tof_diagnostics, clusters_df, period_ns, plots_dir)

    # -------------------------------------------------------------------------
    # Stage F: labeling
    # -------------------------------------------------------------------------
    from tpx3pipe.labeling import label_clusters, plot_labeling_diagnostics
    clusters_df, model = _step("Stage F (labeling)", label_clusters, clusters_df, period_ns, cfg)
    _step("Stage F (plots)", plot_labeling_diagnostics, clusters_df, model, period_ns, plots_dir)

    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)

    # -------------------------------------------------------------------------
    # Stage H: GBT baseline
    # -------------------------------------------------------------------------
    from tpx3pipe.baseline import train_baseline
    baseline_scores, _, clusters_df = _step(
        "Stage H (GBT)", train_baseline, clusters_df, cfg, out_dir, plots_dir
    )
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    print(f"  GBT test accuracy: {baseline_scores['test_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Stage I: CNN (optional)
    # -------------------------------------------------------------------------
    if not args.skip_cnn:
        try:
            from tpx3pipe.cnn.train import train_cnn
            cnn_scores = _step(
                "Stage I (CNN)", train_cnn,
                clusters_df, cluster_pixels_df, cfg, out_dir, plots_dir,
            )
            clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
            print(f"  CNN test accuracy: {cnn_scores['test_accuracy']:.4f}")
        except ImportError as e:
            log.warning("Skipping CNN (torch not installed): %s", e)

    # -------------------------------------------------------------------------
    # Stage G: diagnostics + reports
    # -------------------------------------------------------------------------
    from tpx3pipe.diagnostics import (
        baseline_under_peak, morphology_bimodality,
        confident_disagreement, validate_against_truth, generate_report,
    )

    bl = _step("Stage G (baseline-under-peak)", baseline_under_peak, clusters_df, period_ns)
    bio = _step("Stage G (bimodality)", morphology_bimodality, clusters_df, plots_dir)
    dis = confident_disagreement(clusters_df)
    validation = validate_against_truth(clusters_df, truth_df, bl)
    _step(
        "Stage G (report)", generate_report,
        bl, bio, dis, validation, plots_dir, out_dir / "diagnostics_report.md",
    )

    # -------------------------------------------------------------------------
    # Final results report
    # -------------------------------------------------------------------------
    _write_results_report(out_dir, baseline_scores, validation)
    print(f"\nPipeline complete. Artifacts in {out_dir}/")


def _write_results_report(out_dir: Path, baseline_scores: dict, validation: dict):
    import json
    cnn_path = out_dir / "cnn_scores.json"
    cnn_scores = {}
    if cnn_path.exists():
        with open(cnn_path) as f:
            cnn_scores = json.load(f)

    lines = [
        "# Results Report",
        "",
        "> **Language note:** all labels refer to folded-ToF bins, not particle species.",
        "",
        "## GBT Baseline (Stage H)",
        f"- Test accuracy: {baseline_scores.get('test_accuracy', 'N/A'):.4f}",
        f"- n_train: {baseline_scores.get('n_train', 'N/A')}",
        f"- n_test: {baseline_scores.get('n_test', 'N/A')}",
        "",
    ]
    if cnn_scores:
        lines += [
            "## CNN (Stage I)",
            f"- Test accuracy: {cnn_scores.get('test_accuracy', 'N/A'):.4f}",
            f"- Parameters: {cnn_scores.get('n_params', 'N/A')}",
            f"- Temperature: {cnn_scores.get('temperature', 'N/A'):.4f}",
            "",
        ]
        gbt_acc = baseline_scores.get("test_accuracy", 0)
        cnn_acc = cnn_scores.get("test_accuracy", 0)
        if cnn_acc > gbt_acc + 0.01:
            lines.append(f"CNN outperforms GBT by {cnn_acc - gbt_acc:.4f}.\n")
        else:
            lines.append("CNN does not clearly beat GBT — this is a valid finding.\n")

    if validation:
        lines += [
            "## Label Contamination Validation (Stage G)",
            f"- True contamination fraction: {validation.get('true_contamination_frac', 'N/A'):.4f}",
            f"- Estimated contamination fraction: {validation.get('estimated_contamination_frac', 'N/A'):.4f}",
            f"- Absolute error: {validation.get('error', 'N/A'):.4f}",
            "",
        ]

    lines += [
        "## Open Questions for SMEs",
        "",
        "- **Flight path and trigger offset**: not available — species-to-ToF mapping impossible.",
        "- **Source pulse structure and expected species**: deferred to SME.",
        "- **Energy calibration**: unavailable; raw ToT only.",
        "- **Time-walk correction**: raw timing used; dt_min_vs_argmax diagnostic provided.",
        "- **Expected maximum ToF**: determines fold-period choice — consult with SMEs.",
        "",
    ]

    report_path = out_dir / "results_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Results report written to %s", report_path)
    print(f"Results report: {report_path}")


if __name__ == "__main__":
    main()
