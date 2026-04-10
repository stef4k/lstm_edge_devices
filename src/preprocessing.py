"""
preprocessing.py
Sliding-window segmentation and feature preparation for CSI LSTM.

Pipeline (mirroring the paper):
1. Identify non-zero subcarriers from the entire dataset (hardware artefact:
   some subcarriers at the band edges and DC are always 0).
2. For each recording, apply a sliding window (configurable size / stride).
3. Assemble a 3-D tensor  X  of shape (n_windows, window_samples, n_features)
   and a label vector  y.
4. Normalization is done *inside each cross-validation fold* (z-score per
   subcarrier, fit only on the training split) to avoid data leakage.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ── subcarrier mask ──────────────────────────────────────────────────────────

def compute_nonzero_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Return a boolean mask of shape (n_subcarriers,) marking subcarriers that
    are non-zero in *at least one* recording segment.
    """
    all_csi = [np.asarray(row["csi_amplitude"], dtype=np.float32)
               for _, row in df.iterrows()]
    stacked = np.concatenate(all_csi, axis=0)   # (total_samples, n_subcarriers)
    mask = np.any(stacked != 0, axis=0)
    n_kept = int(mask.sum())
    print(f"[preprocessing] Non-zero subcarriers: {n_kept} / {len(mask)}")
    return mask


# ── sliding window ───────────────────────────────────────────────────────────

def _windows_from_recording(
    csi: np.ndarray,
    label: str,
    record_id: str,
    experiment_id: str,
    window_samples: int,
    stride_samples: int,
    nonzero_mask: np.ndarray,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Segment one recording into overlapping windows.

    Returns
    -------
    X_seg  : (n_windows, window_samples, n_features)  float32
    labels : list of str, length n_windows
    rids   : list of str, length n_windows
    expids : list of str, length n_windows
    """
    csi = np.asarray(csi, dtype=np.float32)
    # Select valid subcarriers
    csi = csi[:, nonzero_mask]
    n_samples, n_features = csi.shape

    if n_samples < window_samples:
        return np.empty((0, window_samples, n_features), np.float32), [], [], []

    n_windows = (n_samples - window_samples) // stride_samples + 1
    X_seg = np.empty((n_windows, window_samples, n_features), dtype=np.float32)
    for w in range(n_windows):
        start = w * stride_samples
        X_seg[w] = csi[start : start + window_samples]

    labels = [label] * n_windows
    rids = [record_id] * n_windows
    expids = [experiment_id] * n_windows
    return X_seg, labels, rids, expids


def build_windows(
    df: pd.DataFrame,
    window_samples: int,
    stride_samples: int,
    nonzero_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Apply sliding window to every recording in *df* and return a flat
    DataFrame with one row per window.

    Columns: window_id, record_id, experiment_id, label, csi_window (ndarray)
    """
    rows = []
    global_idx = 0
    for _, rec in df.iterrows():
        csi = np.asarray(rec["csi_amplitude"], dtype=np.float32)
        X_seg, labels, rids, expids = _windows_from_recording(
            csi,
            str(rec["label"]),
            str(rec["record_id"]),
            str(rec["experiment_id"]),
            window_samples,
            stride_samples,
            nonzero_mask,
        )
        for w in range(len(labels)):
            rows.append(
                {
                    "window_id": f"win_{global_idx:06d}",
                    "record_id": rids[w],
                    "experiment_id": expids[w],
                    "label": labels[w],
                    "csi_window": X_seg[w],
                }
            )
            global_idx += 1

    df_wins = pd.DataFrame(rows)
    print(f"[preprocessing] Total windows: {len(df_wins)}")
    if not df_wins.empty:
        vc = df_wins["label"].value_counts()
        for lbl, cnt in vc.items():
            print(f"  {lbl:<12} {cnt}")
    return df_wins


# ── normalization (fold-safe) ────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-subcarrier mean and std from training windows.

    X_train : (n_windows, window_samples, n_features)
    Returns (mean, std) each of shape (1, 1, n_features).
    """
    flat = X_train.reshape(-1, X_train.shape[-1])   # (n*T, F)
    mean = flat.mean(axis=0, keepdims=True)[None]    # (1, 1, F)
    std = flat.std(axis=0, keepdims=True)[None]      # (1, 1, F)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalize X using pre-computed mean/std."""
    return ((X - mean) / std).astype(np.float32)


# ── window DataFrame → numpy ─────────────────────────────────────────────────

def windows_to_arrays(
    df_wins: pd.DataFrame,
    label_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the window DataFrame to numpy arrays.

    Returns
    -------
    X   : (n_windows, window_samples, n_features)  float32
    y   : (n_windows,)  int  (class indices, order = label_names)
    exp : (n_windows,)  str  (experiment_id per window)
    """
    X = np.stack(df_wins["csi_window"].to_numpy(), axis=0).astype(np.float32)
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in df_wins["label"]], dtype=np.int64)
    exp = df_wins["experiment_id"].to_numpy()
    return X, y, exp
