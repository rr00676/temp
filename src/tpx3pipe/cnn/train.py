"""CNN training, evaluation, saliency, and reliability diagram (Stage I).

Usage::

    python -m tpx3pipe.cnn.train --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, optim, DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for Stage I. "
            "Install via: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


CNN_SCALAR_COLS = ["n_pixels", "tot_sum", "tot_max", "eccentricity"]


def _build_tensors(df: pd.DataFrame, patches: np.ndarray, patch_ids: np.ndarray, use_soft: bool):
    """Align patch array to df by cluster_id; return (patch_tensor, scalar_tensor, label_tensor)."""
    torch, nn, optim, DataLoader, TensorDataset = _require_torch()

    id_to_idx = {cid: i for i, cid in enumerate(patch_ids)}
    indices = [id_to_idx[cid] for cid in df["cluster_id"].values if cid in id_to_idx]
    df_aligned = df[df["cluster_id"].isin(id_to_idx)].copy()

    patch_arr = patches[[id_to_idx[cid] for cid in df_aligned["cluster_id"].values]]

    scalar_arr = df_aligned[CNN_SCALAR_COLS].fillna(0).values.astype(np.float32)

    if use_soft:
        posterior_cols = [c for c in df_aligned.columns if c.startswith("posterior_")]
        if posterior_cols:
            label_arr = df_aligned[posterior_cols].fillna(0).values.astype(np.float32)
        else:
            label_arr = df_aligned["hard_label"].values.astype(np.int64)
            use_soft = False
    else:
        label_arr = df_aligned["hard_label"].values.astype(np.int64)

    patch_t = torch.from_numpy(patch_arr).float()
    scalar_t = torch.from_numpy(scalar_arr).float()
    if use_soft and label_arr.ndim == 2:
        label_t = torch.from_numpy(label_arr).float()
    else:
        label_t = torch.from_numpy(label_arr).long()

    return patch_t, scalar_t, label_t, df_aligned["cluster_id"].values, use_soft


def train_cnn(
    clusters_df: pd.DataFrame,
    cluster_pixels_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    plots_dir: Path,
) -> dict:
    torch, nn, optim, DataLoader, TensorDataset = _require_torch()
    from tpx3pipe.cnn.patches import extract_patches, dihedral_augment
    from tpx3pipe.cnn.model import build_model, TemperatureScaler
    from tpx3pipe.baseline import time_block_split

    ccfg = cfg.get("cnn", {})
    P = int(ccfg.get("patch_size", 32))
    batch_size = int(ccfg.get("batch_size", 64))
    max_epochs = int(ccfg.get("max_epochs", 50))
    patience = int(ccfg.get("early_stopping_patience", 7))
    lr = float(ccfg.get("learning_rate", 1e-3))
    use_soft = bool(ccfg.get("use_soft_labels", True))
    do_aug = bool(ccfg.get("augmentation", True))
    seed = int(cfg.get("seed", 42))
    dt_clip = float(cfg.get("clustering", {}).get("dt_ns", 2000.0))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # filter labeled events
    valid = clusters_df["hard_label"].notna() & (clusters_df["hard_label"] >= 0)
    df_labeled = clusters_df[valid].copy()

    train_df, val_df, test_df = time_block_split(
        df_labeled,
        float(cfg.get("baseline", {}).get("train_frac", 0.70)),
        float(cfg.get("baseline", {}).get("val_frac", 0.15)),
    )

    # extract patches (train determines channel stats)
    train_patches, train_ids, ch_stats = extract_patches(
        cluster_pixels_df, train_df, patch_size=P, dt_clip_ns=dt_clip,
    )
    val_patches, val_ids, _ = extract_patches(
        cluster_pixels_df, val_df, patch_size=P, dt_clip_ns=dt_clip, channel_stats=ch_stats,
    )
    test_patches, test_ids, _ = extract_patches(
        cluster_pixels_df, test_df, patch_size=P, dt_clip_ns=dt_clip, channel_stats=ch_stats,
    )

    # build tensors
    Xp_tr, Xs_tr, y_tr, _, use_soft_actual = _build_tensors(train_df, train_patches, train_ids, use_soft)
    Xp_val, Xs_val, y_val, _, _ = _build_tensors(val_df, val_patches, val_ids, use_soft_actual)
    Xp_te, Xs_te, y_te, te_ids, _ = _build_tensors(test_df, test_patches, test_ids, False)

    n_classes = int(df_labeled["hard_label"].max()) + 1 + 1  # +1 for background
    # infer from posterior columns if available
    post_cols = [c for c in df_labeled.columns if c.startswith("posterior_")]
    if post_cols:
        n_classes = len(post_cols)

    model = build_model(n_classes=n_classes, n_scalar=len(CNN_SCALAR_COLS), patch_size=P)
    log.info("CNN: %d parameters", model.n_params())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    if use_soft_actual:
        criterion = nn.CrossEntropyLoss()   # works with soft targets via manual loop
    else:
        criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(Xp_tr, Xs_tr, y_tr)

    best_val_loss = np.inf
    no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(train_dataset))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xp, xs, yt = Xp_tr[idx], Xs_tr[idx], y_tr[idx]

            if do_aug:
                aug = np.stack([
                    dihedral_augment(xp[i].numpy(), rng) for i in range(len(xp))
                ])
                xp = torch.from_numpy(aug).float()

            logits = model(xp, xs)
            if use_soft_actual and yt.dim() == 2:
                # soft label cross-entropy: -sum(p * log_softmax(logits))
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -(yt * log_probs).sum(dim=-1).mean()
            else:
                loss = criterion(logits, yt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(Xp_val, Xs_val)
            if use_soft_actual and y_val.dim() == 2:
                log_probs = torch.log_softmax(val_logits, dim=-1)
                val_loss = -(y_val * log_probs).sum(dim=-1).mean().item()
            else:
                val_loss = criterion(val_logits, y_val).item()

        scheduler.step(val_loss)
        train_loss = total_loss / max(n_batches, 1)
        log.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            import copy
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # temperature scaling
    ts = TemperatureScaler()
    model.eval()
    with torch.no_grad():
        val_logits_for_ts = model(Xp_val, Xs_val)
    val_hard = y_val.argmax(dim=1) if y_val.dim() == 2 else y_val
    T = ts.fit(val_logits_for_ts.detach(), val_hard)
    log.info("Temperature scaling T=%.4f", T)

    # test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = ts.scale(model(Xp_te, Xs_te))
        test_probs = torch.softmax(test_logits, dim=-1).numpy()
    test_preds = test_probs.argmax(axis=1)
    test_true = y_te.numpy() if y_te.dim() == 1 else y_te.argmax(dim=1).numpy()

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(test_true, test_preds, output_dict=True, zero_division=0)
    log.info("CNN test:\n%s", classification_report(test_true, test_preds, zero_division=0))

    # saliency (input-gradient) on a few samples per class
    _saliency_plot(model, Xp_te, Xs_te, test_true, n_classes, plots_dir)

    # reliability diagram
    _reliability_diagram(test_probs, test_true, plots_dir)

    # confusion matrix
    cm = confusion_matrix(test_true, test_preds)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (ToF label)")
    ax.set_title("CNN confusion matrix (test block)")
    classes = sorted(np.unique(np.concatenate([test_true, test_preds])))
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
    fig.savefig(plots_dir / "cnn_confusion.png", dpi=120)
    plt.close(fig)

    scores = {
        "model": "CNN",
        "n_params": model.n_params(),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "temperature": T,
        "test_accuracy": float((test_preds == test_true).mean()),
        "classification_report": report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cnn_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    # save model
    torch.save({"model_state": model.state_dict(), "temperature": T,
                "channel_stats": ch_stats, "n_classes": n_classes},
               out_dir / "cnn_model.pt")
    log.info("CNN model saved to %s", out_dir / "cnn_model.pt")

    return scores


import matplotlib.pyplot as plt


def _saliency_plot(model, Xp, Xs, labels, n_classes, plots_dir: Path):
    """Input-gradient saliency maps for one sample per class."""
    try:
        torch, nn, *_ = _require_torch()
        plots_dir.mkdir(parents=True, exist_ok=True)
        unique_classes = np.unique(labels)[:n_classes]
        fig, axes = plt.subplots(2, len(unique_classes), figsize=(4 * len(unique_classes), 7))
        if len(unique_classes) == 1:
            axes = axes.reshape(2, 1)

        for col, cls in enumerate(unique_classes):
            idx = np.where(labels == cls)[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            xp = Xp[i:i+1].clone().requires_grad_(True)
            xs = Xs[i:i+1].detach()
            logits = model(xp, xs)
            logits[0, int(cls)].backward()
            saliency = xp.grad.data.abs().squeeze(0).numpy()   # (2, P, P)

            # show ch0 patch and saliency (ch0)
            axes[0, col].imshow(Xp[i, 0].numpy(), cmap="inferno")
            axes[0, col].set_title(f"Class {int(cls)}\nlog1p(ToT)")
            axes[0, col].axis("off")
            axes[1, col].imshow(saliency[0], cmap="hot")
            axes[1, col].set_title("Saliency (ch0)")
            axes[1, col].axis("off")

        fig.suptitle("Input-gradient saliency per class")
        fig.tight_layout()
        fig.savefig(plots_dir / "cnn_saliency.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        log.warning("Saliency plot failed: %s", e)


def _reliability_diagram(probs: np.ndarray, true_labels: np.ndarray, plots_dir: Path, n_bins=10):
    """Calibration reliability diagram."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    max_probs = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(max_probs[mask].mean())
            bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(bin_conf, bin_acc, "o-", color="steelblue", label="CNN")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (test block)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cnn_reliability.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage I: train CNN classifier")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")
    cluster_pixels_df = pd.read_parquet(out_dir / "cluster_pixels.parquet")

    scores = train_cnn(clusters_df, cluster_pixels_df, cfg, out_dir, plots_dir)
    print(f"CNN test accuracy: {scores['test_accuracy']:.4f}")
    print(f"Scores written to {out_dir}/cnn_scores.json")


if __name__ == "__main__":
    main()
