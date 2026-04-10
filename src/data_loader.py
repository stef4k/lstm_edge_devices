"""
data_loader.py
Load and merge ESP32 Wi-Fi CSI recordings from PKL files.

Each PKL contains a DataFrame with one row per continuous recording segment.
We keep only rows whose label belongs to TARGET_LABELS, drop transition rows
("Changing zone"), and tag every row with its experiment ID.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_label(value: object) -> str:
    s = str(value).strip().lower()
    s = s.replace("_", " ")
    return " ".join(s.split())


def _as_2d(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _infer_fs(row: pd.Series, n_samples: int) -> float:
    ts = row.get("timestamps", None)
    if isinstance(ts, (list, tuple, np.ndarray)) and len(ts) == n_samples > 1:
        t = pd.to_numeric(pd.Series(ts), errors="coerce").to_numpy(float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            return float(1.0 / np.median(dt))
    fs = row.get("sampling_freq", np.nan)
    try:
        fs = float(fs)
        if np.isfinite(fs) and fs > 0:
            return fs
    except Exception:
        pass
    return np.nan


def _load_pkl(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        for key in ("df", "data", "dataset"):
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key].copy()
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported payload type for {path}: {type(obj)}")


# ── public API ───────────────────────────────────────────────────────────────

def load_experiments(
    base_dir: str,
    experiments: List[Dict],
    target_labels: List[str],
    transition_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and merge all experiment PKL files.

    Parameters
    ----------
    base_dir : str
        Root directory containing the experiment sub-folders.
    experiments : list of dict
        Each dict has keys ``name`` (str) and ``file`` (str, path relative to
        ``base_dir``).
    target_labels : list of str
        Normalised label strings to keep.  All others (including transitions)
        are dropped.
    transition_labels : list of str, optional
        Labels that mark transition segments; dropped before windowing.

    Returns
    -------
    pd.DataFrame
        One row per recording segment, columns:
        ``record_id, experiment_id, label, csi_amplitude, sampling_freq_hz,
        n_samples, n_subcarriers``
    """
    base = Path(base_dir)
    if transition_labels is None:
        transition_labels = []
    transition_set = {_normalize_label(t) for t in transition_labels}
    target_set = {_normalize_label(t) for t in target_labels}

    parts = []
    for exp in experiments:
        path = base / exp["file"]
        df = _load_pkl(path)
        df = df.copy()
        df["experiment_id"] = exp["name"]
        df["source_file"] = path.name
        parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)
    df_all.insert(0, "record_id", [f"rec_{i:04d}" for i in range(len(df_all))])

    # Use 'label' column (fall back to 'action')
    if "label" not in df_all.columns and "action" in df_all.columns:
        df_all["label"] = df_all["action"]

    df_all["label_norm"] = df_all["label"].apply(_normalize_label)

    # Drop transition rows
    df_all = df_all[~df_all["label_norm"].isin(transition_set)].copy()

    # Keep only target activity rows
    df_all = df_all[df_all["label_norm"].isin(target_set)].copy()

    # Normalised label as a proper Categorical in the requested order
    df_all["label"] = pd.Categorical(
        df_all["label_norm"], categories=target_labels, ordered=True
    )

    # Derived metadata
    df_all["n_samples"] = df_all["csi_amplitude"].apply(
        lambda x: _as_2d(x).shape[0]
    )
    df_all["n_subcarriers"] = df_all["csi_amplitude"].apply(
        lambda x: _as_2d(x).shape[1]
    )
    df_all["sampling_freq_hz"] = df_all.apply(
        lambda r: _infer_fs(r, int(r["n_samples"])), axis=1
    )

    keep_cols = [
        "record_id", "experiment_id", "source_file",
        "label", "label_norm",
        "csi_amplitude",
        "sampling_freq_hz", "n_samples", "n_subcarriers",
    ]
    df_all = df_all[[c for c in keep_cols if c in df_all.columns]].copy()
    df_all = df_all.reset_index(drop=True)

    print(f"[data_loader] Loaded {len(df_all)} recording segments")
    print(f"[data_loader] Label distribution:")
    for label, count in df_all["label"].value_counts().reindex(target_labels).items():
        print(f"  {label:<12} {count}")
    print(f"[data_loader] Experiments: {sorted(df_all['experiment_id'].unique())}")

    return df_all
