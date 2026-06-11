"""Stage H — GBT baseline classifier.

Trains LightGBM on Stage C scalar features with time-block splits (no event-level
random splits).  Reports per-class precision/recall/F1, confusion matrix, and feature
importances.  Scores are written to artifacts/baseline_scores.json for comparison
with the CNN.

Usage::

    python -m tpx3pipe.baseline --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from tpx3pipe.features import SCALAR_FEATURE_COLS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-block split (shared with CNN stage — no temporal overlap)
# ---------------------------------------------------------------------------

def time_block_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    time_col: str = "t_min",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test by contiguous time blocks.

    Events are sorted by time_col and partitioned at cumulative-fraction boundaries.
    No random shuffling — this prevents leakage across the temporal correlation
    structure of the detector.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_block_split: column '{time_col}' not in DataFrame")
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:i_train].copy()
    val = df_sorted.iloc[i_train:i_val].copy()
    test = df_sorted.iloc[i_val:].copy()

    # enforce: no temporal overlap
    if len(train) > 0 and len(val) > 0:
        assert train[time_col].max() <= val[time_col].min(), \
            "Time-block split produced temporal overlap between train and val!"
    if len(val) > 0 and len(test) > 0:
        assert val[time_col].max() <= test[time_col].min(), \
            "Time-block split produced temporal overlap between val and test!"

    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# GBT training
# ---------------------------------------------------------------------------

def _get_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError("lightgbm is required for Stage H. Install via: pip install lightgbm")


def train_baseline(
    clusters_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    plots_dir: Path,
) -> Dict:
    """Train LightGBM classifier; return score dict."""
    lgb = _get_lgbm()

    # filter: only labeled, unambiguous events
    valid = (
        clusters_df["hard_label"].notna()
        & (clusters_df["hard_label"] >= 0)
    )
    df = clusters_df[valid].copy()
    df["hard_label"] = df["hard_label"].astype(int)

    bcfg = cfg.get("baseline", {})
    train_frac = float(bcfg.get("train_frac", 0.70))
    val_frac = float(bcfg.get("val_frac", 0.15))

    train, val, test = time_block_split(df, train_frac, val_frac)

    feat_cols = [c for c in SCALAR_FEATURE_COLS if c in df.columns]
    log.info("Training on features: %s", feat_cols)

    X_train, y_train = train[feat_cols].values, train["hard_label"].values
    X_val, y_val = val[feat_cols].values, val["hard_label"].values
    X_test, y_test = test[feat_cols].values, test["hard_label"].values

    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "random_state": int(cfg.get("seed", 42)),
        "verbose": -1,
    }
    lgbm_params.update(bcfg.get("lgbm_params", {}))

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    log.info("GBT test report:\n%s", classification_report(y_test, y_pred, zero_division=0))

    # feature importances plot
    plots_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_cols[i] for i in order], importances[order], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("GBT feature importances")
    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_importances.png", dpi=120)
    plt.close(fig)

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (ToF label)")
    ax.set_title("GBT confusion matrix (test block)")
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_confusion.png", dpi=120)
    plt.close(fig)

    scores = {
        "model": "LightGBM",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "features": feat_cols,
        "classification_report": report_dict,
        "test_accuracy": float((y_pred == y_test).mean()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    log.info("Baseline scores written to %s", out_dir / "baseline_scores.json")

    # add predictions to clusters_df subset
    df_full = clusters_df.copy()
    df_full["predicted_label"] = np.nan
    pred_all = model.predict(df_full.loc[valid, feat_cols].fillna(0).values)
    df_full.loc[valid, "predicted_label"] = pred_all

    return scores, model, df_full


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage H: GBT baseline classifier")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")

    scores, model, clusters_with_pred = train_baseline(clusters_df, cfg, out_dir, plots_dir)
    clusters_with_pred.to_parquet(out_dir / "clusters.parquet", index=False)

    print(f"GBT test accuracy: {scores['test_accuracy']:.4f}")
    print(f"Scores written to {out_dir}/baseline_scores.json")


if __name__ == "__main__":
    main()
