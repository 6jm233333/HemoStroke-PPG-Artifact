from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.analysis.operating_point import apply_binary_threshold, validate_threshold
from src.models.lstm import LSTMClassifier
from src.models.resnet1d import ResNet1D, resnet1d18


# =============================================================================
# Optional YAML support
# =============================================================================

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class HorizonSpec:
    name: str
    minutes: int
    internal_dir: Path
    external_dir: Optional[Path]
    output_dir: Path


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    n_splits: int = 5
    random_state: int = 42

    positive_label: int = 1
    negative_label: int = 0
    ignore_label_value: int = -1
    drop_ignore_label: bool = True

    monitor_metric: str = "macro_f1"  # paper-aligned
    positive_class_weight: float = 3.0  # Positive-class weight used in the paper.

    sampling_enabled: bool = True
    sampling_mode: str = "weighted_sampler"   # weighted_sampler / none
    target_pos_neg_ratio: float = 1.0
    apply_sampler_to_train_only: bool = True

    save_fold_metrics_csv: bool = True
    save_mean_std_csv: bool = True
    save_predictions_csv: bool = True
    save_confusion_matrix: bool = True
    threshold: float = 0.7910

    external_validation_enabled: bool = True
    retrain_on_full_internal: bool = False
    save_external_predictions: bool = True

    num_workers: int = 0
    pin_memory: bool = True


# =============================================================================
# Utility functions
# =============================================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def deep_get(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, but --config was provided.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_monitor_metric(metric_name: str) -> str:
    x = str(metric_name).strip().lower()
    aliases = {
        "val_f1": "macro_f1",
        "f1": "f1",
        "macro_f1": "macro_f1",
        "recall": "recall",
        "auc": "auc",
        "auprc": "auprc",
        "accuracy": "accuracy",
        "acc": "accuracy",
    }
    return aliases.get(x, "macro_f1")


# =============================================================================
# Array loading
# =============================================================================

def _load_triplet(root_dir: Path, prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_path = root_dir / f"{prefix}_data.npy"
    y_path = root_dir / f"{prefix}_label.npy"
    p_path = root_dir / f"{prefix}_pid.npy"

    if not (x_path.exists() and y_path.exists() and p_path.exists()):
        return None

    x = np.load(x_path, allow_pickle=True).astype(np.float32)
    y = np.load(y_path, allow_pickle=True).astype(np.int64)
    p = np.load(p_path, allow_pickle=True)

    if len(x) != len(y) or len(x) != len(p):
        raise ValueError(
            f"Length mismatch in {root_dir} / {prefix}: "
            f"len(x)={len(x)}, len(y)={len(y)}, len(pid)={len(p)}"
        )

    return x, y, p


def load_merged_arrays(root_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge any available {train,val,test}_*.npy into one dataset.

    This keeps compatibility with the earlier script that concatenated all
    existing arrays and then re-split them with StratifiedGroupKFold.
    """
    parts = []
    for prefix in ("train", "val", "test"):
        triplet = _load_triplet(root_dir, prefix)
        if triplet is not None:
            parts.append(triplet)

    if not parts:
        raise FileNotFoundError(
            f"No valid train/val/test npy triplets found under: {root_dir}"
        )

    x_all = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
    y_all = np.concatenate([p[1] for p in parts], axis=0).astype(np.int64)
    pid_all = np.concatenate([p[2] for p in parts], axis=0)

    return x_all, y_all, pid_all


def filter_labels(
    x: np.ndarray,
    y: np.ndarray,
    pid: np.ndarray,
    *,
    positive_label: int,
    negative_label: int,
    drop_ignore_label: bool,
    ignore_label_value: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if drop_ignore_label:
        mask = y != ignore_label_value
        x = x[mask]
        y = y[mask]
        pid = pid[mask]

    keep_mask = np.isin(y, [negative_label, positive_label])
    x = x[keep_mask]
    y = y[keep_mask]
    pid = pid[keep_mask]

    if len(x) == 0:
        raise ValueError("No valid binary samples remain after label filtering.")

    return x, y, pid


# =============================================================================
# Sampler
# =============================================================================

def build_weighted_sampler(
    y: np.ndarray,
    target_pos_neg_ratio: float = 1.0,
    positive_label: int = 1,
) -> WeightedRandomSampler:
    y = np.asarray(y)
    n_pos = int(np.sum(y == positive_label))
    n_neg = int(np.sum(y != positive_label))

    if n_pos == 0 or n_neg == 0:
        weights = np.ones(len(y), dtype=np.float64)
        return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)

    # target_pos_neg_ratio = desired pos / neg sampling ratio
    neg_weight = 1.0
    pos_weight = (n_neg / max(n_pos, 1)) * float(target_pos_neg_ratio)

    sample_weights = np.where(y == positive_label, pos_weight, neg_weight).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


# =============================================================================
# Model / optimizer
# =============================================================================

def build_model(
    *,
    input_dim: int,
    output_dim: int,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model_cfg = model_cfg or {}
    name = str(model_cfg.get("name", "resnet1d")).lower()
    dropout = float(model_cfg.get("dropout", 0.0))

    if name in {"resnet1d", "resnet"}:
        return resnet1d18(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            input_layout="btc",
        )

    if name == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=dropout,
            bidirectional=bool(model_cfg.get("bidirectional", False)),
        )

    raise ValueError(f"Unsupported model name: {name}")


def build_loss(
    *,
    device: torch.device,
    positive_class_weight: float,
) -> nn.Module:
    # Binary classification mapped to classes [0, 1].
    class_weights = torch.tensor([1.0, float(positive_class_weight)], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# =============================================================================
# Metrics
# =============================================================================

def safe_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, prob_pos))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, prob_pos))
    except Exception:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_pos: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": safe_auc(y_true, prob_pos),
        "auprc": safe_auprc(y_true, prob_pos),
    }


def metric_for_selection(metrics: Dict[str, float], monitor_metric: str) -> float:
    key = infer_monitor_metric(monitor_metric)
    return float(metrics.get(key, float("nan")))


# =============================================================================
# Dataloaders
# =============================================================================

def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler],
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# =============================================================================
# Training / evaluation loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = len(xb)
        loss_sum += float(loss.item()) * bs
        n_samples += bs

    return loss_sum / max(n_samples, 1)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, np.ndarray]:
    model.eval()

    all_logits: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = apply_binary_threshold(probs[:, 1].detach().cpu().numpy(), threshold)

        all_logits.append(logits.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds)
        all_trues.append(yb.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)
    trues_np = np.concatenate(all_trues, axis=0)

    return {
        "logits": logits_np,
        "probs": probs_np,
        "preds": preds_np,
        "trues": trues_np,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()

    loss_sum = 0.0
    n_samples = 0

    pred_bundle = predict_loader(model, loader, device=device, threshold=threshold)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = len(xb)
        loss_sum += float(loss.item()) * bs
        n_samples += bs

    y_true = pred_bundle["trues"]
    y_pred = pred_bundle["preds"]
    prob_pos = pred_bundle["probs"][:, 1]

    metrics = compute_metrics(y_true, y_pred, prob_pos)
    metrics["loss"] = loss_sum / max(n_samples, 1)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "metrics": metrics,
        "pred_bundle": pred_bundle,
    }


# =============================================================================
# Fit helpers
# =============================================================================

def fit_with_early_stopping(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    input_dim = int(x_train.shape[-1])
    model = build_model(input_dim=input_dim, output_dim=2, model_cfg=model_cfg).to(device)
    criterion = build_loss(device=device, positive_class_weight=cfg.positive_class_weight)
    optimizer = build_optimizer(model, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    sampler = None
    if cfg.sampling_enabled and cfg.sampling_mode == "weighted_sampler":
        sampler = build_weighted_sampler(
            y_train,
            target_pos_neg_ratio=cfg.target_pos_neg_ratio,
            positive_label=cfg.positive_label,
        )

    train_loader = build_loader(
        x_train,
        y_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )
    val_loader = build_loader(
        x_val,
        y_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    best_score = -float("inf")
    best_eval: Optional[Dict[str, Any]] = None

    epochs_without_improvement = 0
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_eval = evaluate_loader(model, val_loader, criterion, device, threshold=cfg.threshold)
        val_metrics = val_eval["metrics"]

        score = metric_for_selection(val_metrics, cfg.monitor_metric)
        scheduler.step(score)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_auc": val_metrics["auc"],
                "val_auprc": val_metrics["auprc"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = (
            score > best_score
            or (
                math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
                and val_metrics["recall"] > (best_eval["metrics"]["recall"] if best_eval is not None else -float("inf"))
            )
        )

        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_eval = copy.deepcopy(val_eval)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            break

    if best_state is None or best_eval is None:
        raise RuntimeError("Training failed to produce a valid best model state.")

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_eval": best_eval,
        "history": pd.DataFrame(history_rows),
    }


def fit_fixed_epochs(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    cfg: TrainConfig,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
) -> nn.Module:
    input_dim = int(x_train.shape[-1])
    model = build_model(input_dim=input_dim, output_dim=2, model_cfg=model_cfg).to(device)
    criterion = build_loss(device=device, positive_class_weight=cfg.positive_class_weight)
    optimizer = build_optimizer(model, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    sampler = None
    if cfg.sampling_enabled and cfg.sampling_mode == "weighted_sampler":
        sampler = build_weighted_sampler(
            y_train,
            target_pos_neg_ratio=cfg.target_pos_neg_ratio,
            positive_label=cfg.positive_label,
        )

    train_loader = build_loader(
        x_train,
        y_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    for _ in range(int(max(1, epochs))):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    return model


# =============================================================================
# Output helpers
# =============================================================================

def predictions_to_dataframe(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_pos: np.ndarray,
    pid: np.ndarray,
    fold: Optional[int],
    split_name: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "y_prob": prob_pos.astype(float),
            "pid": pid,
            "split": split_name,
        }
    )
    if fold is not None:
        df["fold"] = int(fold)
    return df


def save_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    save_dataframe(df, path)


# =============================================================================
# Cross-validation
# =============================================================================

def run_internal_cv(
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    pid_all: np.ndarray,
    cfg: TrainConfig,
    model_cfg: Optional[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "histories")

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    fold_rows: List[Dict[str, Any]] = []
    prediction_parts: List[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(x_all, y_all, groups=pid_all), start=1):
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]
        pid_val = pid_all[val_idx]

        fit_res = fit_with_early_stopping(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
            model_cfg=model_cfg,
            device=device,
        )

        model = fit_res["model"]
        history_df = fit_res["history"]
        best_epoch = fit_res["best_epoch"]

        # save checkpoint
        ckpt_path = output_dir / "checkpoints" / f"fold_{fold_idx}_best.pth"
        torch.save(model.state_dict(), ckpt_path)

        # save history
        save_dataframe(history_df, output_dir / "histories" / f"fold_{fold_idx}_history.csv")

        # fold metrics
        best_eval = fit_res["best_eval"]
        metrics = best_eval["metrics"]
        pred_bundle = best_eval["pred_bundle"]

        fold_rows.append(
            {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "macro_f1": metrics["macro_f1"],
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "n_train_patients": int(len(np.unique(pid_all[train_idx]))),
                "n_val_patients": int(len(np.unique(pid_val))),
                "checkpoint_path": str(ckpt_path),
            }
        )

        if cfg.save_predictions_csv:
            prediction_parts.append(
                predictions_to_dataframe(
                    y_true=pred_bundle["trues"],
                    y_pred=pred_bundle["preds"],
                    prob_pos=pred_bundle["probs"][:, 1],
                    pid=pid_val,
                    fold=fold_idx,
                    split_name="internal_val",
                )
            )

        if cfg.save_confusion_matrix:
            save_confusion_matrix_csv(
                metrics["confusion_matrix"],
                output_dir / f"fold_{fold_idx}_confusion_matrix.csv",
            )

    fold_df = pd.DataFrame(fold_rows)
    if cfg.save_fold_metrics_csv:
        save_dataframe(fold_df, output_dir / "fold_metrics.csv")

    summary_numeric = ["loss", "accuracy", "precision", "recall", "f1", "macro_f1", "auc", "auprc"]
    summary_rows = []
    for col in summary_numeric:
        if col not in fold_df.columns:
            continue
        summary_rows.append(
            {
                "metric": col,
                "mean": float(fold_df[col].mean()),
                "std": float(fold_df[col].std(ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if cfg.save_mean_std_csv:
        save_dataframe(summary_df, output_dir / "cv_mean_std_summary.csv")

    if prediction_parts and cfg.save_predictions_csv:
        pred_df = pd.concat(prediction_parts, ignore_index=True)
        save_dataframe(pred_df, output_dir / "internal_cv_predictions.csv")
    else:
        pred_df = pd.DataFrame()

    return {
        "fold_df": fold_df,
        "summary_df": summary_df,
        "pred_df": pred_df,
    }


# =============================================================================
# External validation
# =============================================================================

def run_external_validation(
    *,
    x_internal: np.ndarray,
    y_internal: np.ndarray,
    pid_internal: np.ndarray,
    x_external: np.ndarray,
    y_external: np.ndarray,
    pid_external: np.ndarray,
    cfg: TrainConfig,
    model_cfg: Optional[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(output_dir / "external")

    criterion = build_loss(device=device, positive_class_weight=cfg.positive_class_weight)
    external_loader = build_loader(
        x_external,
        y_external,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    if cfg.retrain_on_full_internal:
        sgkf = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )
        first_train_idx, first_val_idx = next(iter(sgkf.split(x_internal, y_internal, groups=pid_internal)))

        fit_res = fit_with_early_stopping(
            x_train=x_internal[first_train_idx],
            y_train=y_internal[first_train_idx],
            x_val=x_internal[first_val_idx],
            y_val=y_internal[first_val_idx],
            cfg=cfg,
            model_cfg=model_cfg,
            device=device,
        )
        selected_epochs = int(fit_res["best_epoch"])
        final_model = fit_fixed_epochs(
            x_train=x_internal,
            y_train=y_internal,
            epochs=selected_epochs,
            cfg=cfg,
            model_cfg=model_cfg,
            device=device,
        )
        retrain_mode = "full_internal_fixed_epochs"

        eval_res = evaluate_loader(final_model, external_loader, criterion, device, threshold=cfg.threshold)
        metrics = eval_res["metrics"]
        pred_bundle = eval_res["pred_bundle"]

        model_path = output_dir / "external" / "external_best_model.pth"
        torch.save(final_model.state_dict(), model_path)

        metrics_df = pd.DataFrame(
            [
                {
                    "fold": "full_internal",
                    "retrain_mode": retrain_mode,
                    "selected_epochs": selected_epochs,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "auc": metrics["auc"],
                    "auprc": metrics["auprc"],
                    "n_samples": int(len(y_external)),
                    "n_patients": int(len(np.unique(pid_external))),
                    "checkpoint_path": str(model_path),
                }
            ]
        )

        if cfg.save_external_predictions:
            pred_df = predictions_to_dataframe(
                y_true=pred_bundle["trues"],
                y_pred=pred_bundle["preds"],
                prob_pos=pred_bundle["probs"][:, 1],
                pid=pid_external,
                fold=None,
                split_name="external_test",
            )
        else:
            pred_df = pd.DataFrame()

        confusion_for_export = metrics["confusion_matrix"]
    else:
        checkpoint_paths = sorted((output_dir / "checkpoints").glob("fold_*_best.pth"))
        if not checkpoint_paths:
            raise RuntimeError(
                f"No fold-trained checkpoints found under {output_dir / 'checkpoints'}. "
                "Run internal cross-validation before frozen external testing."
            )

        fold_rows: List[Dict[str, Any]] = []
        fold_prediction_parts: List[pd.DataFrame] = []
        fold_probs: List[np.ndarray] = []

        for ckpt_path in checkpoint_paths:
            match = os.path.basename(str(ckpt_path)).split("_")
            fold_id = int(match[1]) if len(match) > 1 and match[1].isdigit() else len(fold_rows) + 1

            model = build_model(
                input_dim=int(x_external.shape[2]),
                output_dim=2,
                model_cfg=model_cfg,
            ).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)

            eval_res = evaluate_loader(model, external_loader, criterion, device, threshold=cfg.threshold)
            metrics = eval_res["metrics"]
            pred_bundle = eval_res["pred_bundle"]
            fold_probs.append(pred_bundle["probs"])

            fold_rows.append(
                {
                    "fold": fold_id,
                    "retrain_mode": "fold_trained_frozen_model",
                    "selected_epochs": np.nan,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "auc": metrics["auc"],
                    "auprc": metrics["auprc"],
                    "n_samples": int(len(y_external)),
                    "n_patients": int(len(np.unique(pid_external))),
                    "checkpoint_path": str(ckpt_path),
                }
            )

            if cfg.save_external_predictions:
                fold_prediction_parts.append(
                    predictions_to_dataframe(
                        y_true=pred_bundle["trues"],
                        y_pred=pred_bundle["preds"],
                        prob_pos=pred_bundle["probs"][:, 1],
                        pid=pid_external,
                        fold=fold_id,
                        split_name="external_test_fold_model",
                    )
                )

        mean_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
        mean_preds = apply_binary_threshold(mean_probs[:, 1], cfg.threshold)
        ensemble_metrics = compute_metrics(y_external, mean_preds, mean_probs[:, 1])
        ensemble_metrics["loss"] = float(np.mean([row["loss"] for row in fold_rows]))
        confusion_for_export = confusion_matrix(y_external, mean_preds, labels=[0, 1])

        aggregate_row = {
            "fold": "ensemble",
            "retrain_mode": "fold_trained_frozen_models_mean_probability",
            "selected_epochs": np.nan,
            "loss": ensemble_metrics["loss"],
            "accuracy": ensemble_metrics["accuracy"],
            "precision": ensemble_metrics["precision"],
            "recall": ensemble_metrics["recall"],
            "f1": ensemble_metrics["f1"],
            "macro_f1": ensemble_metrics["macro_f1"],
            "auc": ensemble_metrics["auc"],
            "auprc": ensemble_metrics["auprc"],
            "n_samples": int(len(y_external)),
            "n_patients": int(len(np.unique(pid_external))),
            "checkpoint_path": "fold_trained_checkpoints",
            "n_fold_models": int(len(checkpoint_paths)),
        }
        metrics_df = pd.concat(
            [pd.DataFrame([aggregate_row]), pd.DataFrame(fold_rows)],
            ignore_index=True,
        )

        if cfg.save_external_predictions:
            pred_df = predictions_to_dataframe(
                y_true=y_external,
                y_pred=mean_preds,
                prob_pos=mean_probs[:, 1],
                pid=pid_external,
                fold=None,
                split_name="external_test_fold_ensemble",
            )
            if fold_prediction_parts:
                pred_df = pd.concat([pred_df] + fold_prediction_parts, ignore_index=True)
        else:
            pred_df = pd.DataFrame()

        selected_epochs = np.nan
        retrain_mode = "fold_trained_frozen_models_mean_probability"

    save_dataframe(metrics_df, output_dir / "external" / "external_metrics.csv")

    if cfg.save_external_predictions:
        save_dataframe(pred_df, output_dir / "external" / "external_predictions.csv")

    if cfg.save_confusion_matrix:
        save_confusion_matrix_csv(
            confusion_for_export,
            output_dir / "external" / "external_confusion_matrix.csv",
        )

    return {
        "metrics_df": metrics_df,
        "pred_df": pred_df,
        "selected_epochs": selected_epochs,
        "retrain_mode": retrain_mode,
    }


# =============================================================================
# Config parsing
# =============================================================================

def build_train_config(cfg_dict: Dict[str, Any]) -> TrainConfig:
    project_seed = int(deep_get(cfg_dict, ["project", "seed"], 42))
    training_cfg = deep_get(cfg_dict, ["training"], {}) or {}
    split_cfg = deep_get(cfg_dict, ["split"], {}) or {}
    sampling_cfg = deep_get(cfg_dict, ["sampling"], {}) or {}
    eval_cfg = deep_get(cfg_dict, ["evaluation"], {}) or {}
    external_cfg = deep_get(cfg_dict, ["external_validation"], {}) or {}
    data_cfg = deep_get(cfg_dict, ["data"], {}) or {}

    pos_weight = deep_get(training_cfg, ["positive_class_weight"], None)
    if pos_weight is None:
        pos_weight = deep_get(cfg_dict, ["training", "lambda_pos"], 3.0)
    if pos_weight is None:
        pos_weight = 3.0

    return TrainConfig(
        seed=project_seed,
        device=str(training_cfg.get("device", "cuda")),
        batch_size=int(training_cfg.get("batch_size", 128)),
        epochs=int(training_cfg.get("epochs", 100)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 15)),
        scheduler_patience=int(training_cfg.get("scheduler_patience", 5)),
        scheduler_factor=float(training_cfg.get("scheduler_factor", 0.5)),
        n_splits=int(split_cfg.get("n_splits", 5)),
        random_state=int(split_cfg.get("random_state", 42)),
        positive_label=int(deep_get(data_cfg, ["labels_to_use", "positive"], 1)),
        negative_label=int(deep_get(data_cfg, ["labels_to_use", "negative"], 0)),
        ignore_label_value=int(data_cfg.get("ignore_label_value", -1)),
        drop_ignore_label=bool(data_cfg.get("drop_ignore_label", True)),
        monitor_metric=infer_monitor_metric(training_cfg.get("monitor_metric", "val_f1")),
        positive_class_weight=float(pos_weight),
        sampling_enabled=bool(sampling_cfg.get("enabled", True)),
        sampling_mode=str(sampling_cfg.get("mode", "weighted_sampler")),
        target_pos_neg_ratio=float(sampling_cfg.get("target_pos_neg_ratio", 1.0)),
        apply_sampler_to_train_only=bool(sampling_cfg.get("apply_to_train_only", True)),
        save_fold_metrics_csv=bool(eval_cfg.get("save_fold_metrics_csv", True)),
        save_mean_std_csv=bool(eval_cfg.get("save_mean_std_csv", True)),
        save_predictions_csv=bool(eval_cfg.get("save_predictions_csv", True)),
        save_confusion_matrix=bool(eval_cfg.get("save_confusion_matrix", True)),
        threshold=validate_threshold(float(eval_cfg.get("threshold", 0.7910))),
        external_validation_enabled=bool(external_cfg.get("enabled", True)),
        retrain_on_full_internal=bool(external_cfg.get("retrain_on_full_internal", False)),
        save_external_predictions=bool(external_cfg.get("save_external_predictions", True)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
    )


def build_horizon_specs(cfg_dict: Dict[str, Any]) -> List[HorizonSpec]:
    horizons = deep_get(cfg_dict, ["experiments", "horizons"], None)
    if not horizons:
        raise ValueError("No experiments.horizons found in config.")

    specs: List[HorizonSpec] = []
    for h in horizons:
        name = str(h["name"])
        minutes = int(h.get("minutes", h.get("hours", 0) * 60))
        internal_dir = Path(h["train_data_dir"]["mimic"])
        external_dir = Path(h["train_data_dir"]["mcmed"]) if "mcmed" in h.get("train_data_dir", {}) else None
        output_dir = Path(h["output_dir"])
        specs.append(
            HorizonSpec(
                name=name,
                minutes=minutes,
                internal_dir=internal_dir,
                external_dir=external_dir,
                output_dir=output_dir,
            )
        )
    return specs


# =============================================================================
# Main runner
# =============================================================================

def run_single_horizon(
    *,
    spec: HorizonSpec,
    cfg: TrainConfig,
    full_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    device = to_device(cfg.device)
    seed_everything(cfg.seed)

    model_cfg = deep_get(full_cfg, ["model"], {}) or {}

    x_internal, y_internal, pid_internal = load_merged_arrays(spec.internal_dir)
    x_internal, y_internal, pid_internal = filter_labels(
        x_internal,
        y_internal,
        pid_internal,
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
        drop_ignore_label=cfg.drop_ignore_label,
        ignore_label_value=cfg.ignore_label_value,
    )

    run_output_dir = spec.output_dir
    ensure_dir(run_output_dir)

    internal_res = run_internal_cv(
        x_all=x_internal,
        y_all=y_internal,
        pid_all=pid_internal,
        cfg=cfg,
        model_cfg=model_cfg,
        output_dir=run_output_dir,
        device=device,
    )

    external_res = None
    if cfg.external_validation_enabled and spec.external_dir is not None and spec.external_dir.exists():
        x_external, y_external, pid_external = load_merged_arrays(spec.external_dir)
        x_external, y_external, pid_external = filter_labels(
            x_external,
            y_external,
            pid_external,
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
            drop_ignore_label=cfg.drop_ignore_label,
            ignore_label_value=cfg.ignore_label_value,
        )

        external_res = run_external_validation(
            x_internal=x_internal,
            y_internal=y_internal,
            pid_internal=pid_internal,
            x_external=x_external,
            y_external=y_external,
            pid_external=pid_external,
            cfg=cfg,
            model_cfg=model_cfg,
            output_dir=run_output_dir,
            device=device,
        )

    manifest = {
        "horizon_name": spec.name,
        "minutes": spec.minutes,
        "internal_dir": str(spec.internal_dir),
        "external_dir": str(spec.external_dir) if spec.external_dir is not None else None,
        "output_dir": str(run_output_dir),
        "device": str(device),
        "n_internal_samples": int(len(y_internal)),
        "n_internal_patients": int(len(np.unique(pid_internal))),
        "n_internal_positive": int(np.sum(y_internal == cfg.positive_label)),
        "n_internal_negative": int(np.sum(y_internal == cfg.negative_label)),
        "n_splits": cfg.n_splits,
        "monitor_metric": cfg.monitor_metric,
        "positive_class_weight": cfg.positive_class_weight,
        "operating_threshold": cfg.threshold,
    }

    if external_res is not None and spec.external_dir is not None:
        x_external, y_external, pid_external = load_merged_arrays(spec.external_dir)
        x_external, y_external, pid_external = filter_labels(
            x_external,
            y_external,
            pid_external,
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
            drop_ignore_label=cfg.drop_ignore_label,
            ignore_label_value=cfg.ignore_label_value,
        )
        manifest.update(
            {
                "n_external_samples": int(len(y_external)),
                "n_external_patients": int(len(np.unique(pid_external))),
                "n_external_positive": int(np.sum(y_external == cfg.positive_label)),
                "n_external_negative": int(np.sum(y_external == cfg.negative_label)),
            }
        )

    save_json(manifest, run_output_dir / "run_manifest.json")

    return {
        "manifest": manifest,
        "internal": internal_res,
        "external": external_res,
    }


def summarize_across_horizons(results: List[Dict[str, Any]], output_root: Path) -> None:
    rows = []
    for res in results:
        manifest = res["manifest"]
        internal_summary = res["internal"]["summary_df"]
        ext_res = res["external"]

        row = {
            "horizon_name": manifest["horizon_name"],
            "minutes": manifest["minutes"],
        }

        for _, r in internal_summary.iterrows():
            row[f"internal_{r['metric']}_mean"] = r["mean"]
            row[f"internal_{r['metric']}_std"] = r["std"]

        if ext_res is not None:
            ext_metrics = ext_res["metrics_df"].iloc[0].to_dict()
            for k in ["accuracy", "precision", "recall", "f1", "macro_f1", "auc", "auprc"]:
                row[f"external_{k}"] = ext_metrics.get(k, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    save_dataframe(df, output_root / "all_horizon_summary.csv")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured classifier for pre-anchor stroke risk-state classification.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--single-internal-dir",
        type=str,
        default=None,
        help="Optional override for a single internal data directory.",
    )
    parser.add_argument(
        "--single-external-dir",
        type=str,
        default=None,
        help="Optional override for a single external data directory.",
    )
    parser.add_argument(
        "--single-output-dir",
        type=str,
        default=None,
        help="Optional override for a single output directory.",
    )
    parser.add_argument(
        "--single-name",
        type=str,
        default="single_run",
        help="Run name when using --single-* overrides.",
    )
    parser.add_argument(
        "--single-minutes",
        type=int,
        default=0,
        help="Minutes label for the single run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg_dict = load_yaml_config(cfg_path)
    train_cfg = build_train_config(cfg_dict)

    if args.single_internal_dir is not None:
        spec = HorizonSpec(
            name=str(args.single_name),
            minutes=int(args.single_minutes),
            internal_dir=Path(args.single_internal_dir),
            external_dir=Path(args.single_external_dir) if args.single_external_dir else None,
            output_dir=Path(args.single_output_dir) if args.single_output_dir else Path("outputs") / str(args.single_name),
        )
        results = [run_single_horizon(spec=spec, cfg=train_cfg, full_cfg=cfg_dict)]
        summarize_across_horizons(results, spec.output_dir.parent)
        print(f"Finished single run: {spec.name}")
        return

    specs = build_horizon_specs(cfg_dict)
    results = []
    for spec in specs:
        print("=" * 88)
        print(f"Running horizon: {spec.name} ({spec.minutes} min)")
        print(f"Internal dir: {spec.internal_dir}")
        print(f"External dir: {spec.external_dir}")
        print(f"Output dir  : {spec.output_dir}")
        res = run_single_horizon(spec=spec, cfg=train_cfg, full_cfg=cfg_dict)
        results.append(res)

    common_root = Path("outputs") / "main"
    summarize_across_horizons(results, common_root)
    print("=" * 88)
    print("Training finished.")


if __name__ == "__main__":
    main()
