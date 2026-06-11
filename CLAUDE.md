# CLAUDE.md — Timepix3 ToF Cluster Classification Pipeline

## Project overview

Build an offline analysis pipeline for Timepix3 pixel-detector data from a pulsed
multimodal (photon + neutron) radiography beam. The pipeline:

1. Clusters raw pixel hits into particle events (streaming algorithm, Meduna et al.,
   CTD/WIT 2019, arXiv:1910.13356 — adapted with configurable spatial/temporal windows).
2. Computes time-of-flight (ToF) per cluster against a TDC pulse-reference stream.
3. Derives labels by mixture-fitting the **folded** ToF spectrum.
4. Trains classifiers (GBT baseline, then CNN) to predict folded-ToF bin from
   cluster morphology/energy alone.
5. Produces diagnostics quantifying label contamination from ToF wraparound.

**Deliverable framing (important):** the classifier predicts *folded-ToF bin*, not
particle species. Species interpretation is deferred to domain experts. All reports
and docstrings must use this language.

**CRITICAL CONSTRAINT: No real data is available on this system.** All development
must be validated against a synthetic data generator (see §8) that emulates the real
data's schema and known pathologies. Design every stage so that pointing it at real
data later requires only a config change, not code changes.

## 1. Data schema (real data, to arrive later)

Two input dataframes (assume parquet or CSV; make loader format-agnostic):

**Pixel dataframe** — one row per pixel hit:
| column | meaning |
|---|---|
| `x_pix`, `y_pix` | pixel coordinates, 0–255 (256×256 matrix, 55 µm pitch) |
| `toa` | coarse Time of Arrival (chip clock units) |
| `ftoa` | fine ToA |
| `spidr_time` | readout coarse timestamp |
| `time_ns` | **fully corrected global time in ns — use this as the sole time axis** |

**TDC dataframe** — one row per beam-pulse reference edge:
| column | meaning |
|---|---|
| `time_ns` | global time of the TDC edge, ns. Only column we rely on; ignore others. |

Treat `toa`/`ftoa`/`spidr_time` as opaque (rollover handling assumed already done
upstream into `time_ns`). Add a loader-level sanity check that `time_ns` is
monotone-ish and spans a plausible acquisition (~5 s).

## 2. Known physics facts and assumptions (encode as config, never hardcode)

- Pulse period: **~1780–1790 ns** (user's numbers approximate). The pipeline must
  **measure** the period from the TDC stream (median of edge diffs) and use the
  measured value everywhere. Config may supply an override.
- Folded ToF spectrum structure (approximate, from user's recollection):
  flat low baseline 0–~150 ns → prominent peak ~175 ns (likely gamma flash) →
  secondary peak ~250 ns → long tail to the period cutoff.
- Empirically, folding mod the period does not smear → working in the folded
  coordinate is legitimate, **and** wraparound contamination is present (the flat
  baseline and tail-to-cutoff are signatures of slow particles from earlier pulses).
- Per-pixel dead time 475 ns; pixel stream may be unsorted within a ~200 µs horizon.
- No energy calibration or time-walk correction is available — raw ToT/ToA only.
- Flight path, trigger offset, and source spectrum are **unknown**. Do not attempt
  to unfold ToF or assign species. List these in the final report's open-questions
  section.

## 3. Engineering standards

- Python 3.11+, `uv` or `pip` with pinned `requirements.txt`. Core deps: numpy,
  pandas, scipy, scikit-learn, matplotlib; lightgbm (or xgboost) for the baseline;
  pytorch for the CNN. Keep torch optional (lazy import) so stages 1–7 run without it.
- Single YAML config (`config.yaml`) drives everything: file paths, clustering
  Δt and connectivity, patch size, bin definitions, split fractions, seeds.
- Deterministic: seed numpy/torch/sklearn from config.
- Package layout: `src/tpx3pipe/` with modules `io.py`, `tdc.py`, `clustering.py`,
  `features.py`, `tof.py`, `labeling.py`, `diagnostics.py`, `baseline.py`,
  `cnn/`, `synth.py`; thin CLI entry points per stage (e.g. `python -m tpx3pipe.cluster
  --config config.yaml`). Each stage reads/writes parquet artifacts in `artifacts/`
  so stages are independently rerunnable.
- pytest suite; every algorithmic claim below has a test against synthetic data.
- Vectorize with numpy where possible; clustering is the only genuinely sequential
  stage. Target: 5 s of beam data (assume up to ~10⁷ pixel rows) clusters in minutes
  on a laptop. Profile before optimizing further.

## 4. Stage A — TDC diagnostics (`tdc.py`)

Run first; everything downstream depends on a trusted TDC stream.

- Compute consecutive diffs of TDC `time_ns`. Report: median (→ measured period),
  spread, and a histogram.
- Flag missed edges: diffs near integer multiples (≥2×) of the median period.
  Report count and fraction of affected periods.
- Sanity: `n_edges × period ≈ total span`; report discrepancy.
- Output: `artifacts/tdc_report.json` + diagnostic plot. The measured period is
  written into the run manifest and consumed by later stages.

## 5. Stage B — Clustering (`clustering.py`)

Streaming open-cluster algorithm (paper's Algorithm 1), with these adaptations:

- **Time axis:** use `time_ns` directly.
- **Sort handling:** sort the pixel stream by `time_ns` up front (offline mode makes
  this cheap and sidesteps the 200 µs unsorted-horizon complexity). Keep the
  open-cluster close-out logic correct regardless: a cluster closes when the stream
  time passes `cluster_max_time + Δt + safety_margin`.
- **Temporal condition (configurable):** pixel joins a cluster if its `time_ns` is
  within `Δt` of the cluster's most recent pixel (chain criterion) — config option
  to switch to "within Δt of cluster min" (paper's stricter pairwise reading).
  Default Δt = 2000 ns per the paper; expose for sweeps (e.g. 200–2000 ns).
- **Spatial condition (configurable):** default 8-connectivity; config option
  `max_gap` allowing pixels within Chebyshev distance `1 + max_gap` (gap-tolerant
  connectivity) for the "wider connectivity" variant.
- **Implementation:** use a dict/spatial-hash of recently active pixels keyed by
  (x, y) instead of a quadtree — simpler and adequate offline. Cluster merge via
  union-find or small-set transfer.
- **Outputs:** `clusters.parquet` (one row per cluster: id, n_pixels, t_min,
  t_max, summary features from Stage C) and `cluster_pixels.parquet`
  (pixel→cluster assignment) for patch rendering.
- **Required diagnostic:** parameter-sensitivity sweep over (Δt, max_gap) reporting
  cluster count, size distribution, and merge/split indicators. Plot.

Tests: synthetic streams with known ground-truth clusters, including (a) two clusters
near-coincident in time but spatially separate, (b) one cluster with a deliberately
late low-ToT pixel (time-walk emulation), (c) out-of-order pixel arrival, (d) clusters
spanning a close-out boundary.

## 6. Stage C — Cluster features (`features.py`)

Per cluster compute: `n_pixels`, `tot_sum`, `tot_max`, `tot_mean`,
energy-weighted centroid, bounding-box width/height, eccentricity + linearity from
PCA of pixel coordinates (ToT-weighted), perimeter-ish/compactness proxy,
`t_min` (min `time_ns`), `t_argmax_tot` (time of max-ToT pixel),
`dt_min_vs_argmax = t_argmax_tot − t_min` (time-walk indicator).

Diagnostic: distribution of `dt_min_vs_argmax` overall and vs `n_pixels` — if it
diverges systematically for a subpopulation, flag in the report (time-walk artifact).

## 7. Stage D — ToF and folding (`tof.py`)

- Cluster time: configurable — `t_min` (default) or `t_argmax_tot`. Compute both
  and report their ToF-spectrum difference as a diagnostic.
- Raw ToF: cluster time minus the most recent TDC edge ≤ it (searchsorted).
- Folded ToF: `raw_tof mod measured_period` (these coincide unless edges were
  missed; report how often they differ — this measures missed-edge impact).
- Outputs: folded-ToF column appended to clusters; 1D folded spectrum plot;
  **ToF × ToT bivariate histogram** plot (user observed periodicity structure
  there — reproduce it).

## 8. Stage E — Synthetic data generator (`synth.py`)  ← build early, everything tests against this

Generate (pixel_df, tdc_df) matching the real schema, from a config specifying:

- Pulse train: period (~1785 ns), jitter, total duration, optional missed-edge rate.
- Species list, each with: rate per pulse, true-ToF distribution (allow means
  larger than the period → wraparound is generated naturally, giving ground-truth
  contamination), and a cluster-morphology archetype:
  - **dot** — 1–4 pixels, low ToT (photon/electron-like)
  - **curl** — random-walk track, 5–25 pixels, low–mid ToT (electron-like)
  - **blob** — compact round, high central ToT with falloff (heavy recoil-like)
- Per-pixel emission: position from archetype shape, ToT from archetype profile,
  `time_ns` = pulse time + true ToF + small intra-cluster spread + an extra delay
  on low-ToT pixels (time-walk emulation). Optionally shuffle emission order within
  a 200 µs horizon.
- Ground-truth columns (cluster id, species, true ToF, wrap count) saved separately —
  **never readable by pipeline stages**, only by tests and diagnostics-validation.

This gives end-to-end ground truth: clustering accuracy, fold correctness, mixture-fit
label purity vs true purity, and classifier ceiling can all be verified.

## 9. Stage F — Labeling via mixture fit (`labeling.py`)

- Fit the folded-ToF spectrum with K Gaussians + **one uniform background
  component** (the wrapped baseline). K from config; default 2 (≈175 ns, ≈250 ns
  peaks) + background. Fit via EM on event-level ToFs (sklearn GMM won't do the
  uniform component — implement a small custom EM or fit the histogram; either is
  acceptable, test against synthetic truth).
- Per event: posterior probability vector over (peak₁, peak₂, …, background) →
  **soft labels**.
- Hard labels: argmax posterior, with events below a config confidence threshold
  (default 0.9) marked `ambiguous` and excluded from training (kept for evaluation).
- Report per-bin **estimated contamination**: background posterior mass within each
  peak's ToF range. This is the label-noise ceiling — state it prominently.

## 10. Stage G — Contamination diagnostics (`diagnostics.py`)

The deliverable's credibility rests here. Implement:

1. **Baseline-under-peak estimate:** uniform level from the pre-first-peak region;
   baseline-to-peak ratio per bin → mislabel fraction per class.
2. **Within-bin morphology bimodality:** for each ToF bin, plot distributions of
   `n_pixels`, `tot_sum`, `tot_max`, eccentricity; run a 1–2 component GMM /
   dip-style check per feature; flag bins with clear bimodality and estimate the
   minority fraction.
3. (After classifiers exist) **Confident-disagreement map:** events where the
   classifier confidently predicts a different bin than the ToF label, broken down
   by ToF bin and predicted class. Concentrated, consistent disagreement = measured
   wraparound contamination. Validate the whole chain on synthetic data where true
   contamination is known.

Output: a single `diagnostics_report.md` with embedded figures.

## 11. Stage H — GBT baseline (`baseline.py`)

- LightGBM (or xgboost) on Stage C scalar features only.
- **Split by contiguous time blocks** (default 70/15/15 by acquisition time).
  Random event-level splits are forbidden — enforce in the split utility, shared
  with the CNN stage.
- Class weights for imbalance. Report per-class precision/recall/F1, confusion
  matrix vs held-out folded-ToF labels, and feature importances.
- This is the bar the CNN must beat; record its scores in the run manifest.

## 12. Stage I — CNN (`cnn/`)

- **Patches:** dense P×P (default 32; verify ≥99% of clusters fit, else bump to 48)
  centered on the max-ToT pixel. Two channels:
  ch0 = log1p(ToT), ch1 = `time_ns − cluster_t_min` (clipped at Δt). Global
  (train-set) channel scaling; never per-cluster max-normalization.
- **Leakage rules (enforce, with a test):** no TDC-derived quantity, no absolute
  time, no chip (x, y) position in the input tensor.
- **Augmentation:** the 8 dihedral transforms (90° rotations + flips) only.
- **Architecture:** 3–4 conv blocks (conv-BN-ReLU-pool) → global average pool →
  concat with scalar features (`n_pixels`, `tot_sum`, `tot_max`, eccentricity) →
  2-layer MLP head → softmax over bins incl. background. Target ~10⁴–10⁵ params.
- **Loss:** cross-entropy against **soft labels** (Stage F posteriors); config flag
  to fall back to hard labels + ambiguous-exclusion.
- Same time-block splits as Stage H. Early stopping on val loss. CPU-feasible.
- **Post-training:** temperature scaling on the val block; reliability diagram;
  Grad-CAM (or simple input-gradient saliency) on a sample per class — confirm
  attention on the cluster, not padding/corners; confident-disagreement analysis
  feeding Stage G item 3.
- Report side-by-side with GBT. If the CNN does not clearly beat GBT on the held-out
  block, say so plainly — that is a valid finding, not a failure.

## 13. Definition of done (all runnable with zero real data)

- [ ] `python -m tpx3pipe.synth --config config.yaml` produces schema-correct data
      with ground truth, including wraparound contamination.
- [ ] Full pipeline runs end-to-end on synthetic data via a single `make all` or
      `run_all.py`.
- [ ] pytest green; includes: clustering correctness vs ground truth (split/merge
      cases), fold correctness, mixture-fit purity vs true purity within tolerance,
      leakage tests, split-utility test (no temporal overlap).
- [ ] Diagnostics correctly *measure* the synthetic contamination (compare estimated
      vs true mislabel fraction; report the error).
- [ ] `diagnostics_report.md` and a final `results_report.md` generated, using
      folded-ToF-bin language throughout, with an **Open questions for SMEs**
      section: flight path / trigger offset, source pulse structure & species,
      energy calibration & time-walk correction availability, expected max ToF.
- [ ] A short `REAL_DATA.md` describing exactly what to change in `config.yaml`
      when the real files arrive, plus first-run sanity checks to perform
      (TDC report, period value, cluster-rate plot, folded spectrum vs the
      remembered shape: baseline → ~175 ns peak → ~250 ns peak → tail).

## 14. Out of scope

- ToF unfolding / species assignment (needs SME input).
- Real-time/online operation, GUI, energy calibration, time-walk correction.
- Deep architecture search; one compact CNN is sufficient.
