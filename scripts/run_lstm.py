#!/usr/bin/env python3
"""
run_lstm.py
Entry point: load config, run preprocessing + 5-fold LSTM training.

Usage
-----
From the project root:
    python scripts/run_lstm.py
    python scripts/run_lstm.py --config configs/config.yaml
    python scripts/run_lstm.py --config configs/config.yaml --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

# ── make sure the project root is on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.data_loader import load_experiments
from src.evaluate import summarise_cv
from src.preprocessing import (
    build_windows,
    compute_nonzero_mask,
    windows_to_arrays,
)
from src.train import run_cross_validation


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM HAR training — paper replication")
    p.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    # Allow overriding key hyper-parameters from the command line
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None, dest="learning_rate")
    p.add_argument("--lstm_units",   type=int,   default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--window",       type=int,   default=None,
                   help="Window size in samples")
    p.add_argument("--stride",       type=int,   default=None,
                   help="Stride in samples")
    p.add_argument("--output_dir",   default=None,
                   help="Override results directory")
    return p.parse_args()


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI argument overrides into the config dict."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.lstm_units is not None:
        cfg["model"]["lstm_units"] = args.lstm_units
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.window is not None:
        cfg["preprocessing"]["window_samples"] = args.window
    if args.stride is not None:
        cfg["preprocessing"]["stride_samples"] = args.stride
    if args.output_dir is not None:
        cfg["output"]["results_dir"] = args.output_dir
    return cfg


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Change working directory to project root so relative paths work
    os.chdir(PROJECT_ROOT)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Output directory
    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[Step 1] Loading data")
    df_activity = load_experiments(
        base_dir=cfg["data"]["base_dir"],
        experiments=cfg["data"]["experiments"],
        target_labels=cfg["data"]["target_labels"],
        transition_labels=cfg["data"]["transition_labels"],
    )

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n[Step 2] Preprocessing")
    pp = cfg["preprocessing"]
    window_samples = pp["window_samples"]
    stride_samples = pp["stride_samples"]
    print(f"  Window: {window_samples} samples, Stride: {stride_samples} samples")

    nonzero_mask = compute_nonzero_mask(df_activity)
    df_windows = build_windows(
        df_activity,
        window_samples=window_samples,
        stride_samples=stride_samples,
        nonzero_mask=nonzero_mask,
    )

    label_names = cfg["data"]["target_labels"]
    X, y, exp_ids = windows_to_arrays(df_windows, label_names)
    print(f"  X shape: {X.shape}  (n_windows, T, F)")
    print(f"  y shape: {y.shape}")
    print(f"  Label encoding: {dict(enumerate(label_names))}")

    # ── 3. Cross-validation ──────────────────────────────────────────────────
    print("\n[Step 3] 5-fold cross-validation training")
    tr = cfg["training"]
    mdl = cfg["model"]
    cv = cfg["cv"]

    results = run_cross_validation(
        X=X,
        y=y,
        exp_ids=exp_ids,
        label_names=label_names,
        fold_config=cv["folds"],
        empty_test_samples=cv["empty_test_samples"],
        lstm_units=mdl["lstm_units"],
        dropout=mdl["dropout"],
        learning_rate=tr["learning_rate"],
        epochs=tr["epochs"],
        batch_size=tr["batch_size"],
        early_stopping_patience=tr["early_stopping"]["patience"],
        random_seed=cv["random_seed"],
        output_dir=output_dir,
    )

    # ── 4. Evaluation ────────────────────────────────────────────────────────
    print("\n[Step 4] Evaluation")
    summarise_cv(results, label_names, output_dir=output_dir)

    # Save raw results as numpy archive for later analysis
    fold_accs = [r["test_acc"] for r in results]
    np.save(output_dir / "fold_accuracies.npy", np.array(fold_accs))
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
