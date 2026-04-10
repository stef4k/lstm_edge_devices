"""
train.py
5-fold cross-validation training loop — PyTorch implementation.

CV design (mirrors the notebook exactly):
  - 5 activity experiments: exp1, exp2, exp3, exp4, exp8
  - 2 empty-room experiments: exp5_empty, exp6_empty
  - Each fold:
      test  = all windows from one activity exp
              + EMPTY_TEST_SAMPLES windows randomly sampled from one empty exp
      train = windows from the remaining 4 activity exps
              + ALL windows from the other empty exp
      Unsampled windows of the test-role empty exp are EXCLUDED from training.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model
from .preprocessing import apply_scaler, fit_scaler


# ── device ────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── fold construction ─────────────────────────────────────────────────────────

def make_cv_folds(
    exp_ids: np.ndarray,
    fold_config: List[Dict],
    empty_test_samples: int,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Build (train_idx, test_idx, fold_label) for each fold.

    Parameters
    ----------
    exp_ids           : (n_windows,) array of experiment_id strings
    fold_config       : list of dicts with keys activity_test, empty_test,
                        empty_train
    empty_test_samples: number of empty windows to include in the test set
    random_seed       : for reproducibility of empty-window sampling

    Returns
    -------
    list of (train_idx, test_idx, fold_label)
    """
    exp_ids = np.asarray(exp_ids)
    n = len(exp_ids)
    rng = np.random.default_rng(random_seed)
    folds = []
    for fc in fold_config:
        act_exp  = fc["activity_test"]
        test_emp = fc["empty_test"]

        act_idx       = np.where(exp_ids == act_exp)[0]
        test_emp_pool = np.where(exp_ids == test_emp)[0]

        n_sample       = min(empty_test_samples, len(test_emp_pool))
        empty_test_idx = rng.choice(test_emp_pool, size=n_sample, replace=False)

        test_idx = np.concatenate([act_idx, empty_test_idx])

        # Exclude all windows from both the test activity exp and the
        # test-role empty exp (preventing leakage of unsampled windows).
        excluded  = set(act_idx.tolist()) | set(test_emp_pool.tolist())
        train_idx = np.array(
            [i for i in range(n) if i not in excluded], dtype=np.int64
        )

        fold_label = f"test={act_exp}+{test_emp}(n_empty={n_sample})"
        folds.append((train_idx, test_idx, fold_label))

    return folds


# ── training utilities ────────────────────────────────────────────────────────

def _to_tensor(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    # torch.tensor copies data; avoids numpy 2.x / PyTorch 1.12 bridge issue
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return TensorDataset(Xt, yt)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ── single-fold training ──────────────────────────────────────────────────────

def train_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    label_names: List[str],
    lstm_units: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    fold_idx: int,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Train the LSTM on one fold and return evaluation metrics.

    Returns
    -------
    dict with keys: fold, train_acc, test_acc, history,
                    y_true, y_pred, y_pred_prob, label_names
    """
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    # Normalize: fit on training data only to avoid leakage
    mean, std = fit_scaler(X_tr)
    X_tr = apply_scaler(X_tr, mean, std)
    X_te = apply_scaler(X_te, mean, std)

    device     = _get_device()
    n_features = X_tr.shape[2]

    # Data loaders
    train_loader = DataLoader(
        _to_tensor(X_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    test_loader = DataLoader(
        _to_tensor(X_te, y_te),
        batch_size=batch_size, shuffle=False,
    )

    # Model
    model = build_model(
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=lstm_units,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_acc  = -1.0
    best_state    = None
    patience_ctr  = 0
    stopped_epoch = epochs
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _eval_epoch(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"
        )

        # Early stopping
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stopping_patience:
                stopped_epoch = epoch
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save best weights
    if output_dir is not None:
        ckpt_path = output_dir / f"fold_{fold_idx + 1}_best.pt"
        torch.save(best_state, ckpt_path)

    # Final evaluation — use .tolist() to avoid numpy 2.x / PyTorch 1.12 bridge
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(device))
            probs  = torch.softmax(logits, dim=1).cpu()
            all_probs.extend(probs.tolist())

    y_pred_prob = np.array(all_probs, dtype=np.float32)
    y_pred      = y_pred_prob.argmax(axis=1).astype(int)

    _, train_acc = _eval_epoch(model, train_loader, criterion, device)
    _, test_acc  = _eval_epoch(model, test_loader,  criterion, device)

    return {
        "fold": fold_idx + 1,
        "train_acc": float(train_acc),
        "test_acc":  float(test_acc),
        "stopped_epoch": stopped_epoch,
        "elapsed_sec": elapsed,
        "history": history,
        "y_true": y_te.astype(int),
        "y_pred": y_pred.astype(int),
        "y_pred_prob": y_pred_prob,
        "label_names": label_names,
        "mean": mean,
        "std": std,
    }


# ── 5-fold cross-validation ───────────────────────────────────────────────────

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    exp_ids: np.ndarray,
    label_names: List[str],
    fold_config: List[Dict],
    empty_test_samples: int,
    lstm_units: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Run the full 5-fold cross-validation and return per-fold result dicts.
    """
    print(f"Using device: {_get_device()}")
    folds     = make_cv_folds(exp_ids, fold_config, empty_test_samples, random_seed)
    n_classes = len(label_names)
    results   = []

    for fold_idx, (train_idx, test_idx, fold_label) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1} / {len(folds)}: {fold_label}")
        print(f"  train windows: {len(train_idx)}")
        print(f"  test  windows: {len(test_idx)}")

        unique, counts = np.unique(y[train_idx], return_counts=True)
        dist = {label_names[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  train distribution: {dist}")
        unique, counts = np.unique(y[test_idx], return_counts=True)
        dist = {label_names[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  test  distribution: {dist}")

        result = train_fold(
            X=X, y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            n_classes=n_classes,
            label_names=label_names,
            lstm_units=lstm_units,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            fold_idx=fold_idx,
            output_dir=output_dir,
        )
        result["fold_label"] = fold_label
        results.append(result)

        print(
            f"  => test accuracy: {result['test_acc']:.4f}  "
            f"(stopped at epoch {result['stopped_epoch']})"
        )

    return results
