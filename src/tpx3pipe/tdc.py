"""Stage A — TDC diagnostics.

Measures the beam-pulse period from the TDC edge stream and reports data quality.
The measured period is written to artifacts/tdc_report.json and consumed by later
stages via ``load_period()``.

Usage::

    python -m tpx3pipe.tdc --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def run_tdc_diagnostics(tdc_df: pd.DataFrame, cfg: dict, out_dir: Path, plots_dir: Path) -> dict:
    """Compute TDC diagnostics; return report dict (also written as JSON)."""
    times = tdc_df["time_ns"].values
    if len(times) < 2:
        raise ValueError("TDC stream has fewer than 2 edges — cannot measure period.")

    diffs = np.diff(times)

    tdc_cfg = cfg.get("tdc", {})
    override = tdc_cfg.get("period_override_ns")
    missed_thresh_factor = tdc_cfg.get("missed_edge_threshold", 1.5)

    if override is not None:
        measured_period = float(override)
        log.info("Using period override: %.3f ns", measured_period)
    else:
        measured_period = float(np.median(diffs))
        log.info("Measured period (median diff): %.3f ns", measured_period)

    period_spread = float(np.std(diffs))

    # missed edges: gaps near integer multiples ≥ 2 of the period
    multiple = diffs / measured_period
    missed_mask = (multiple >= missed_thresh_factor) & (multiple < 100)
    n_missed = int(missed_mask.sum())
    # estimated number of missing edges in those gaps
    n_missing_edges = int(np.round(multiple[missed_mask] - 1).sum()) if n_missed > 0 else 0
    total_edges = len(times)
    expected_edges = (times[-1] - times[0]) / measured_period + 1
    frac_missed = n_missing_edges / max(expected_edges, 1)

    # sanity: n_edges × period ≈ total span
    total_span = times[-1] - times[0]
    expected_span = (total_edges - 1) * measured_period
    span_discrepancy_ns = abs(total_span - expected_span)

    report = {
        "measured_period_ns": measured_period,
        "period_spread_ns": period_spread,
        "n_edges": total_edges,
        "n_gaps_with_missed_edges": n_missed,
        "n_missing_edges_estimated": n_missing_edges,
        "frac_periods_affected": round(frac_missed, 6),
        "total_span_ns": float(total_span),
        "expected_span_ns": float(expected_span),
        "span_discrepancy_ns": float(span_discrepancy_ns),
    }

    # write JSON report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tdc_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("TDC report written to %s", report_path)

    # diagnostic plot
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(diffs, bins=200, color="steelblue", edgecolor="none")
    ax.axvline(measured_period, color="red", lw=1.5, label=f"median={measured_period:.1f} ns")
    ax.set_xlabel("Inter-edge gap (ns)")
    ax.set_ylabel("Count")
    ax.set_title("TDC inter-edge gap distribution")
    ax.legend()

    ax = axes[1]
    clipped = np.clip(diffs, 0, 5 * measured_period)
    ax.plot(np.arange(len(clipped)), clipped, lw=0.4, color="steelblue")
    ax.axhline(measured_period, color="red", lw=1, label="measured period")
    ax.axhline(missed_thresh_factor * measured_period, color="orange", lw=1,
               linestyle="--", label="missed-edge threshold")
    ax.set_xlabel("Edge index")
    ax.set_ylabel("Gap to next edge (ns)")
    ax.set_title("TDC gap sequence (missed-edge detection)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "tdc_diagnostics.png", dpi=120)
    plt.close(fig)
    log.info("TDC diagnostic plot saved")

    _print_report(report)
    return report


def _print_report(r: dict) -> None:
    print("=== TDC Diagnostics ===")
    print(f"  Measured period:          {r['measured_period_ns']:.3f} ns")
    print(f"  Period spread (std):      {r['period_spread_ns']:.3f} ns")
    print(f"  Total edges:              {r['n_edges']}")
    print(f"  Gaps with missed edges:   {r['n_gaps_with_missed_edges']}")
    print(f"  Estimated missing edges:  {r['n_missing_edges_estimated']}")
    print(f"  Fraction periods affected:{r['frac_periods_affected']:.4f}")
    print(f"  Span discrepancy:         {r['span_discrepancy_ns']:.1f} ns")


def load_period(artifacts_dir: str | Path, cfg: dict) -> float:
    """Return the measured (or overridden) period from the TDC report JSON."""
    override = cfg.get("tdc", {}).get("period_override_ns")
    if override is not None:
        return float(override)
    report_path = Path(artifacts_dir) / "tdc_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"TDC report not found at {report_path}. Run Stage A first."
        )
    with open(report_path) as f:
        return float(json.load(f)["measured_period_ns"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage A: TDC diagnostics")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, load_tdc, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    tdc_df = load_tdc(cfg["paths"]["tdc_data"])
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)
    run_tdc_diagnostics(tdc_df, cfg, out_dir, plots_dir)


if __name__ == "__main__":
    main()
