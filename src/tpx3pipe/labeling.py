"""Stage F — Mixture-fit labeling via custom EM.

Fits K Gaussians + 1 uniform background component to the folded-ToF spectrum.
Returns per-event posterior probability vectors → soft labels.
Hard labels = argmax posterior (events below confidence_threshold → "ambiguous").

The uniform component models wrapped slow particles (contamination from earlier pulses).

Usage::

    python -m tpx3pipe.labeling --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AMBIGUOUS_LABEL = -1


# ---------------------------------------------------------------------------
# Custom EM: K Gaussians + 1 uniform background on [0, period_ns]
# ---------------------------------------------------------------------------

class GaussianUniformMixture:
    """EM fitter for K Gaussians + uniform background on [0, T].

    Parameters
    ----------
    n_peaks : int
        Number of Gaussian components (background is added automatically).
    T : float
        Period (support of the uniform component).
    means_init, stds_init : list[float]
        Initial Gaussian means and stds.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence tolerance.
    """

    def __init__(
        self,
        n_peaks: int,
        T: float,
        means_init: List[float],
        stds_init: List[float],
        n_iter: int = 200,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.n_peaks = n_peaks
        self.T = T
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        assert len(means_init) == n_peaks
        assert len(stds_init) == n_peaks
        self.means_ = np.array(means_init, dtype=float)
        self.stds_ = np.clip(np.array(stds_init, dtype=float), 1.0, None)
        # uniform weight + n_peaks Gaussian weights
        n_comp = n_peaks + 1
        self.weights_ = np.full(n_comp, 1.0 / n_comp)

    @property
    def n_components(self) -> int:
        return self.n_peaks + 1   # background is last

    def _component_densities(self, X: np.ndarray) -> np.ndarray:
        """(N, n_components) density matrix (NOT weighted)."""
        N = len(X)
        D = np.zeros((N, self.n_components))
        for k in range(self.n_peaks):
            mu, sigma = self.means_[k], self.stds_[k]
            D[:, k] = np.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        # uniform component density = 1/T
        D[:, -1] = 1.0 / self.T
        return D

    def fit(self, X: np.ndarray) -> "GaussianUniformMixture":
        X = np.asarray(X, dtype=float)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            D = self._component_densities(X)
            weighted = D * self.weights_[np.newaxis, :]
            row_sum = weighted.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, 1e-300)
            gamma = weighted / row_sum   # (N, K+1)

            # M-step
            Nk = gamma.sum(axis=0)   # (K+1,)
            self.weights_ = Nk / len(X)

            for k in range(self.n_peaks):
                self.means_[k] = (gamma[:, k] * X).sum() / max(Nk[k], 1e-9)
                var = (gamma[:, k] * (X - self.means_[k]) ** 2).sum() / max(Nk[k], 1e-9)
                self.stds_[k] = max(np.sqrt(var), 1.0)

            # log-likelihood
            ll = np.log(row_sum).sum()
            if abs(ll - prev_ll) < self.tol:
                log.debug("EM converged at iteration %d, LL=%.4f", iteration, ll)
                break
            prev_ll = ll

        self.log_likelihood_ = prev_ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, n_components) posterior probabilities."""
        X = np.asarray(X, dtype=float)
        D = self._component_densities(X)
        weighted = D * self.weights_[np.newaxis, :]
        row_sum = weighted.sum(axis=1, keepdims=True)
        return weighted / np.maximum(row_sum, 1e-300)

    def component_names(self) -> List[str]:
        names = [f"peak_{k}" for k in range(self.n_peaks)]
        names.append("background")
        return names


# ---------------------------------------------------------------------------
# Labeling entry point
# ---------------------------------------------------------------------------

def label_clusters(
    clusters_df: pd.DataFrame,
    period_ns: float,
    cfg: dict,
) -> Tuple[pd.DataFrame, GaussianUniformMixture]:
    """Fit mixture model; append posterior columns, hard_label, label_confidence.

    Hard label: index of argmax posterior (background = n_peaks).
    Ambiguous: confidence < threshold → hard_label = AMBIGUOUS_LABEL (-1).
    """
    lcfg = cfg.get("labeling", {})
    n_peaks = int(lcfg.get("n_peaks", 2))
    means_init = list(lcfg.get("peak_means_init_ns", [175.0, 250.0]))[:n_peaks]
    stds_init = list(lcfg.get("peak_stds_init_ns", [20.0, 20.0]))[:n_peaks]
    n_iter = int(lcfg.get("n_em_iter", 200))
    conf_thresh = float(lcfg.get("confidence_threshold", 0.90))
    seed = int(cfg.get("seed", 42))

    valid = clusters_df["folded_tof_ns"].notna()
    X = clusters_df.loc[valid, "folded_tof_ns"].values

    model = GaussianUniformMixture(
        n_peaks=n_peaks,
        T=period_ns,
        means_init=means_init,
        stds_init=stds_init,
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(X)
    log.info(
        "Mixture fit: weights=%s means_ns=%s stds_ns=%s LL=%.2f",
        np.round(model.weights_, 3),
        np.round(model.means_, 1),
        np.round(model.stds_, 1),
        model.log_likelihood_,
    )

    proba = model.predict_proba(X)   # (N_valid, n_components)

    out = clusters_df.copy()
    comp_names = model.component_names()
    for j, name in enumerate(comp_names):
        col = f"posterior_{name}"
        out[col] = np.nan
        out.loc[valid, col] = proba[:, j]

    confidence = proba.max(axis=1)
    hard_label = proba.argmax(axis=1)
    hard_label[confidence < conf_thresh] = AMBIGUOUS_LABEL

    out["label_confidence"] = np.nan
    out["hard_label"] = np.nan
    out.loc[valid, "label_confidence"] = confidence
    out.loc[valid, "hard_label"] = hard_label.astype(float)

    n_ambig = (hard_label == AMBIGUOUS_LABEL).sum()
    log.info(
        "Hard labels: %d total valid, %d ambiguous (%.1f%%), threshold=%.2f",
        len(X), n_ambig, 100 * n_ambig / max(len(X), 1), conf_thresh,
    )
    return out, model


def plot_labeling_diagnostics(
    clusters_df: pd.DataFrame,
    model: GaussianUniformMixture,
    period_ns: float,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    valid = clusters_df["folded_tof_ns"].notna()
    X = clusters_df.loc[valid, "folded_tof_ns"].values

    bins = np.linspace(0, period_ns, 200)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    hist, _ = np.histogram(X, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_centers, hist, width=bin_width, color="steelblue", alpha=0.6, label="data")

    scale = len(X) * bin_width
    t_plot = np.linspace(0, period_ns, 1000)
    D = model._component_densities(t_plot)
    for k in range(model.n_peaks):
        ax.plot(t_plot, D[:, k] * model.weights_[k] * scale,
                lw=2, label=f"peak_{k} (μ={model.means_[k]:.1f}, σ={model.stds_[k]:.1f})")
    ax.axhline(model.weights_[-1] / period_ns * scale, color="gray", lw=1.5,
               linestyle="--", label=f"background (w={model.weights_[-1]:.3f})")
    total = D @ model.weights_ * scale
    ax.plot(t_plot, total, "k-", lw=1.5, label="mixture total")

    ax.set_xlabel("Folded ToF (ns)")
    ax.set_ylabel("Clusters / bin")
    ax.set_title("Mixture model fit to folded-ToF spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "labeling_fit.png", dpi=120)
    plt.close(fig)
    log.info("Labeling fit plot saved")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage F: mixture-fit labeling")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    from tpx3pipe.io import load_config, ensure_artifacts_dir, ensure_plots_dir
    from tpx3pipe.tdc import load_period
    cfg = load_config(args.config)
    out_dir = ensure_artifacts_dir(cfg)
    plots_dir = ensure_plots_dir(cfg)

    clusters_df = pd.read_parquet(out_dir / "clusters.parquet")
    period_ns = load_period(out_dir, cfg)

    clusters_df, model = label_clusters(clusters_df, period_ns, cfg)
    plot_labeling_diagnostics(clusters_df, model, period_ns, plots_dir)
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    print(f"Updated {out_dir}/clusters.parquet with label columns")


if __name__ == "__main__":
    main()
