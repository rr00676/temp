"""Format-agnostic loader for pixel and TDC dataframes.

Supports parquet and CSV.  The only required columns are documented in CLAUDE.md §1.
A loader-level sanity check verifies that ``time_ns`` is monotone-ish and spans a
plausible acquisition duration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

log = logging.getLogger(__name__)

PIXEL_REQUIRED = {"x_pix", "y_pix", "time_ns"}
TDC_REQUIRED = {"time_ns"}

_PLAUSIBLE_SPAN_S = (1e-6, 3600.0)  # 1 µs – 1 h (tests use tiny datasets)
_MONOTONE_FRACTION = 0.99            # at least this fraction of diffs must be ≥ 0


def _load_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Use .parquet or .csv.")


def _check_monotone(df: pd.DataFrame, label: str) -> None:
    diffs = df["time_ns"].diff().dropna()
    frac_ok = (diffs >= 0).mean()
    if frac_ok < _MONOTONE_FRACTION:
        log.warning(
            "%s time_ns is not monotone-ish: %.1f%% of diffs are negative",
            label,
            100 * (1 - frac_ok),
        )
    else:
        log.debug("%s time_ns monotone check passed (%.1f%% non-negative)", label, 100 * frac_ok)


def _check_span(df: pd.DataFrame, label: str) -> None:
    span_s = (df["time_ns"].max() - df["time_ns"].min()) / 1e9
    lo, hi = _PLAUSIBLE_SPAN_S
    if not (lo <= span_s <= hi):
        log.warning(
            "%s time_ns span = %.3f s — outside plausible range [%.1f, %.0f] s",
            label,
            span_s,
            lo,
            hi,
        )
    else:
        log.debug("%s time_ns span = %.3f s (plausible)", label, span_s)


def load_pixels(path: str | Path) -> pd.DataFrame:
    """Load pixel hit dataframe; verify required columns and sanity checks."""
    df = _load_df(path)
    missing = PIXEL_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Pixel dataframe missing required columns: {missing}")
    _check_monotone(df, "pixel")
    _check_span(df, "pixel")
    log.info("Loaded %d pixel hits from %s", len(df), path)
    return df


def load_tdc(path: str | Path) -> pd.DataFrame:
    """Load TDC dataframe; verify required columns and sanity checks."""
    df = _load_df(path)
    missing = TDC_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"TDC dataframe missing required columns: {missing}")
    df = df.sort_values("time_ns").reset_index(drop=True)
    _check_monotone(df, "TDC")
    _check_span(df, "TDC")
    log.info("Loaded %d TDC edges from %s", len(df), path)
    return df


def load_from_tpx3(
    filepath: str | Path,
    module_path: str | None = None,
    column_map: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pixel and TDC dataframes from a real TPX3 file via readtpx3.py.

    Parameters
    ----------
    filepath : path to the .tpx3 (or equivalent) data file
    module_path : directory containing readtpx3.py, or None to use sys.path as-is
    column_map : optional dict mapping the reader's column names to pipeline names,
                 e.g. {"Time": "time_ns", "ToT": "tot"}.  Only needed if the
                 reader returns columns with different names.

    Returns
    -------
    pixel_df, tdc_df — validated and sanity-checked, ready for pipeline stages.
    """
    import importlib.util
    import sys

    if module_path is not None:
        mod_dir = str(Path(module_path).resolve())
        if mod_dir not in sys.path:
            sys.path.insert(0, mod_dir)

    try:
        import readtpx3
    except ImportError as e:
        raise ImportError(
            f"Could not import readtpx3. "
            f"Make sure readtpx3.py is on sys.path (module_path={module_path!r}).\n"
            f"Original error: {e}"
        )

    result = readtpx3.read_tpx3_file(str(filepath))
    if not (isinstance(result, (list, tuple)) and len(result) == 2):
        raise ValueError(
            "readtpx3.read_tpx3_file() must return (pixel_df, tdc_df). "
            f"Got: {type(result)}"
        )
    pixel_df, tdc_df = result

    if column_map:
        pixel_df = pixel_df.rename(columns=column_map)
        tdc_df = tdc_df.rename(columns=column_map)

    # run the same checks as the file-based loaders
    missing_pix = PIXEL_REQUIRED - set(pixel_df.columns)
    if missing_pix:
        raise ValueError(
            f"Pixel dataframe from readtpx3 missing required columns: {missing_pix}. "
            f"Use 'column_map' in config to rename them."
        )
    missing_tdc = TDC_REQUIRED - set(tdc_df.columns)
    if missing_tdc:
        raise ValueError(
            f"TDC dataframe from readtpx3 missing required columns: {missing_tdc}. "
            f"Use 'column_map' in config to rename them."
        )

    tdc_df = tdc_df.sort_values("time_ns").reset_index(drop=True)
    _check_monotone(pixel_df, "pixel (TPX3)")
    _check_span(pixel_df, "pixel (TPX3)")
    _check_monotone(tdc_df, "TDC (TPX3)")
    _check_span(tdc_df, "TDC (TPX3)")

    log.info(
        "Loaded from TPX3: %d pixel hits, %d TDC edges — file: %s",
        len(pixel_df), len(tdc_df), filepath,
    )
    return pixel_df, tdc_df


def load_config(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_artifacts_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["artifacts"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_plots_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["plots"])
    p.mkdir(parents=True, exist_ok=True)
    return p
