# Switching to Real Data

This document describes exactly what to change in `config.yaml` when the real
Timepix3 data files arrive, and which first-run sanity checks to perform.

## 1. Config changes required

### Option A — load via readtpx3.py (recommended for real data)

Open `config.yaml` and set the `tpx3:` block:

```yaml
tpx3:
  file: "/path/to/your/datafile.tpx3"   # path to the real data file
  module_path: "/path/to/dir/containing/readtpx3.py"   # omit if already on PYTHONPATH
  column_map: {}   # leave empty if column names already match (see §2 below)
```

The pipeline calls `readtpx3.read_tpx3_file(file)` automatically.  No other
changes are needed — `paths.pixel_data` and `paths.tdc_data` are ignored when
`tpx3.file` is set.

### Option B — pre-convert to parquet/CSV first

If you prefer to convert once and reuse:

```python
from readtpx3 import read_tpx3_file
pixel_df, tdc_df = read_tpx3_file("your_file.tpx3")
pixel_df.to_parquet("data/pixels.parquet", index=False)
tdc_df.to_parquet("data/tdc.parquet",   index=False)
```

Then leave `tpx3.file: null` in `config.yaml` and set:

```yaml
paths:
  pixel_data: "data/pixels.parquet"
  tdc_data:   "data/tdc.parquet"
```

Everything else is driven by `config.yaml` — no code changes are needed.

### Optional overrides to tune

| Parameter | Location in config | When to change |
|---|---|---|
| `tdc.period_override_ns` | `tdc:` section | Set if the auto-measured period is wrong (check TDC report first) |
| `clustering.dt_ns` | `clustering:` | Lower if too many spurious merges; raise if clusters are fragmented |
| `clustering.max_gap` | `clustering:` | Raise to 1 if clusters have spatial gaps |
| `labeling.n_peaks` | `labeling:` | Adjust if the folded spectrum shows more/fewer Gaussian peaks |
| `labeling.peak_means_init_ns` | `labeling:` | Update initial guesses from your observed folded-ToF spectrum |
| `labeling.confidence_threshold` | `labeling:` | Lower to include more events; raise for cleaner labels |

## 2. Column name requirements

The pipeline requires these column names in the pixel dataframe:

| Required name | Type | Notes |
|---|---|---|
| `x_pix` | int | 0–255 |
| `y_pix` | int | 0–255 |
| `time_ns` | float | Fully corrected global time in ns |
| `tot` | int | Raw ToT value (used for features and CNN) |

Extra columns (`toa`, `ftoa`, `spidr_time`) are accepted and ignored.

If `readtpx3.read_tpx3_file()` returns different column names, list them in
`tpx3.column_map`. Only columns that differ need to be listed:

```yaml
tpx3:
  column_map:
    Time: time_ns      # example: rename "Time" → "time_ns"
    ToT:  tot          # example: rename "ToT"  → "tot"
```

The TDC dataframe only needs `time_ns`.

**TDC file** must contain at minimum:

| Column | Type | Notes |
|---|---|---|
| `time_ns` | float | Global time of each beam-pulse reference edge |

## 3. First-run sanity checks

Run these in order and inspect the outputs before proceeding to later stages.

### Step 1 — TDC diagnostics
```
python -m tpx3pipe.tdc --config config.yaml
```
Check `artifacts/tdc_report.json`:
- `measured_period_ns` should be in the range **1780–1790 ns**.
- `frac_periods_affected` (missed edges) should be small (< 5%).
- `span_discrepancy_ns` should be < 1% of total span.
- Look at `plots/tdc_diagnostics.png` — the gap histogram should show a sharp
  peak at ~1785 ns with a clean single-mode shape.

### Step 2 — Synthetic vs real comparison
Run the synth generator and compare the TDC report to confirm that the pipeline
can measure the period correctly on known-good data:
```
python -m tpx3pipe.synth --config config.yaml
python -m tpx3pipe.tdc --config config.yaml
```

### Step 3 — Clustering spot-check
```
python -m tpx3pipe.cluster --config config.yaml
```
Inspect cluster count and size distribution. Expected signs of a healthy run:
- Cluster rate: roughly consistent with expected particle flux.
- Typical cluster sizes: 1–4 px (dot-like) and 5–30 px (tracks/blobs).
- Very few clusters with >100 pixels (may indicate merge artefact → lower `dt_ns`).

### Step 4 — Folded-ToF spectrum check
```
python -m tpx3pipe.features --config config.yaml
python -m tpx3pipe.tof --config config.yaml
```
Check `plots/tof_spectrum.png`. The folded spectrum should show:
- A flat baseline from 0 to ~150 ns.
- A prominent peak near **~175 ns** (gamma flash).
- A secondary peak near **~250 ns**.
- A tail falling off to the period cutoff.

If the peaks appear at different positions, update `labeling.peak_means_init_ns`
before running Stage F.

### Step 5 — Labeling fit
```
python -m tpx3pipe.labeling --config config.yaml
```
Check `plots/labeling_fit.png` — the mixture model should visually fit the spectrum.
If it does not converge, try adjusting:
- `labeling.peak_means_init_ns`
- `labeling.peak_stds_init_ns`
- `labeling.n_em_iter` (increase to 500 if needed)

## 4. Running the full pipeline
Once sanity checks pass:
```
python run_all.py --config config.yaml --skip-cnn   # stages A–H + diagnostics
python run_all.py --config config.yaml              # full pipeline including CNN
```
Or with make:
```
make no-cnn    # skip CNN
make all       # full pipeline
```
