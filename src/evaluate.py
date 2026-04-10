"""
evaluate.py
Aggregate cross-validation results and produce:
  - Per-fold accuracy table
  - Average accuracy ± std
  - Confusion matrix (aggregated across all folds)
  - Per-class precision / recall / F1 / specificity  (Table I in the paper)
  - Accuracy / loss curves per fold (training history)
  - Saved PNG figures in the output directory
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ── confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = "Aggregated Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ticks = range(len(label_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(label_names, fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved confusion matrix → {save_path}")
    plt.close(fig)


# ── training curves ───────────────────────────────────────────────────────────

def plot_training_history(
    results: List[Dict],
    save_dir: Optional[Path] = None,
) -> None:
    n_folds = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res in results:
        hist = res["history"]
        # PyTorch trainer uses "train_acc"/"val_acc" keys
        tr_key  = "train_acc"  if "train_acc"  in hist else "accuracy"
        val_key = "val_acc"    if "val_acc"    in hist else "val_accuracy"
        epochs = range(1, len(hist[tr_key]) + 1)
        label = f"Fold {res['fold']}"
        axes[0].plot(epochs, hist[tr_key],  alpha=0.7, label=label)
        axes[1].plot(epochs, hist[val_key], alpha=0.7, label=label)

    for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Accuracy per Fold", fontsize=13)
    plt.tight_layout()
    if save_dir is not None:
        path = save_dir / "training_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved training curves → {path}")
    plt.close(fig)


def plot_loss_history(
    results: List[Dict],
    save_dir: Optional[Path] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res in results:
        hist = res["history"]
        tr_key  = "train_loss" if "train_loss" in hist else "loss"
        val_key = "val_loss"
        epochs = range(1, len(hist[tr_key]) + 1)
        label = f"Fold {res['fold']}"
        axes[0].plot(epochs, hist[tr_key],  alpha=0.7, label=label)
        axes[1].plot(epochs, hist[val_key], alpha=0.7, label=label)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Loss per Fold", fontsize=13)
    plt.tight_layout()
    if save_dir is not None:
        path = save_dir / "loss_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved loss curves → {path}")
    plt.close(fig)


# ── per-class metrics (Table I in the paper) ──────────────────────────────────

def _specificity_per_class(
    cm: np.ndarray,
) -> np.ndarray:
    """
    Specificity = TN / (TN + FP) per class, computed from the confusion matrix.
    """
    n = cm.shape[0]
    spec = np.zeros(n, dtype=float)
    for i in range(n):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        denom = tn + fp
        spec[i] = tn / denom if denom > 0 else 0.0
    return spec


def print_metrics_table(
    cm: np.ndarray,
    label_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """
    Print per-class precision, recall, F1, specificity — matching Table I in
    the paper — and overall accuracy.
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        np.repeat(np.arange(len(label_names)), cm.sum(axis=1)),
        np.concatenate([
            np.full(int(cm[i, j]), j)
            for i in range(len(label_names))
            for j in range(len(label_names))
        ]),
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    spec = _specificity_per_class(cm)

    # Macro averages
    macro_prec = precision.mean()
    macro_rec  = recall.mean()
    macro_f1   = f1.mean()
    macro_spec = spec.mean()

    # Overall accuracy (micro)
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    header = f"{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Specificity':>12}"
    sep    = "-" * len(header)
    lines  = [sep, header, sep]
    for i, name in enumerate(label_names):
        lines.append(
            f"{name:<14} {precision[i]:>10.4f} {recall[i]:>8.4f} "
            f"{f1[i]:>8.4f} {spec[i]:>12.4f}"
        )
    lines.append(sep)
    lines.append(
        f"{'Macro Avg':<14} {macro_prec:>10.4f} {macro_rec:>8.4f} "
        f"{macro_f1:>8.4f} {macro_spec:>12.4f}"
    )
    lines.append(sep)
    lines.append(f"Overall Accuracy: {overall_acc:.4f}  ({total_correct}/{total_samples})")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)

    if save_path is not None:
        save_path.write_text(text + "\n")
        print(f"[evaluate] Saved metrics table → {save_path}")


# ── summary ───────────────────────────────────────────────────────────────────

def summarise_cv(
    results: List[Dict],
    label_names: List[str],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Print the per-fold accuracy table, average accuracy, the aggregated
    confusion matrix and the per-class metrics table.

    All plots are saved to *output_dir* if provided.
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    # Per-fold accuracy
    print(f"\n{'Fold':<6} {'Test Experiment':<35} {'Accuracy':>10}")
    print("-" * 54)
    fold_accs = []
    for res in results:
        act_test = res["fold_label"].split("test=")[1].split("+")[0]
        print(f"{res['fold']:<6} {act_test:<35} {res['test_acc']:>10.4f}")
        fold_accs.append(res["test_acc"])

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print("-" * 54)
    print(f"{'Average':<42} {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"\n5-fold CV accuracy: {mean_acc * 100:.1f}% ± {std_acc * 100:.1f}%")

    # Aggregated confusion matrix
    n = len(label_names)
    cm_agg = np.zeros((n, n), dtype=int)
    for res in results:
        cm = confusion_matrix(
            res["y_true"], res["y_pred"], labels=list(range(n))
        )
        cm_agg += cm

    print(f"\nAggregated confusion matrix (all folds):")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in label_names)
    print(header)
    for i, row_name in enumerate(label_names):
        row = f"{row_name:>12}" + "".join(f"{cm_agg[i, j]:>12}" for j in range(n))
        print(row)

    # Per-class metrics
    print("\n")
    print_metrics_table(
        cm_agg,
        label_names,
        save_path=output_dir / "metrics_table.txt" if output_dir else None,
    )

    # Plots
    if output_dir is not None:
        plot_confusion_matrix(
            cm_agg,
            label_names,
            title=f"Aggregated Confusion Matrix — 5-fold CV\n(mean acc {mean_acc*100:.1f}%±{std_acc*100:.1f}%)",
            save_path=output_dir / "confusion_matrix.png",
        )
        plot_training_history(results, save_dir=output_dir)
        plot_loss_history(results, save_dir=output_dir)

    print("\n" + "=" * 60)
