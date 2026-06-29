from __future__ import annotations
import argparse
import json
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
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from src.analysis.operating_point import apply_binary_threshold, validate_threshold
from src.models.lstm import LSTMClassifier
from src.models.resnet1d import resnet1d18


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
class EvalConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 128
    n_splits: int = 5
    random_state: int = 42

    positive_label: int = 1
    negative_label: int = 0
    ignore_label_value: int = -1
    drop_ignore_label: bool = True
    positive_class_weight: float = 3.0
    threshold: float = 0.7910

    save_internal_predictions_csv: bool = True
    save_external_predictions_csv: bool = True
    save_fold_metrics_csv: bool = True
    save_horizon_summary_csv: bool = True
    save_confusion_matrix_csv: bool = True
    save_main_performance_table_csv: bool = True

    num_workers: int = 0
    pin_memory: bool = True


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, but --config was provided.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_mean_std(mean_val: float, std_val: float, digits: int = 4) -> str:
    if pd.isna(mean_val):
        return "nan"
    if pd.isna(std_val):
        return f"{mean_val:.{digits}f} ± nan"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


# =============================================================================
# Config parsing
# =============================================================================

def build_eval_config(cfg_dict: Dict[str, Any]) -> EvalConfig:
    project_seed = int(deep_get(cfg_dict, ["project", "seed"], 42))
    split_cfg = deep_get(cfg_dict, ["split"], {}) or {}
    data_cfg = deep_get(cfg_dict, ["data"], {}) or {}
    eval_cfg = deep_get(cfg_dict, ["evaluation"], {}) or {}
    external_cfg = deep_get(cfg_dict, ["external_validation"], {}) or {}
    training_cfg = deep_get(cfg_dict, ["training"], {}) or {}

    pos_weight = deep_get(training_cfg, ["positive_class_weight"], None)
    if pos_weight is None:
        pos_weight = deep_get(training_cfg, ["lambda_pos"], 3.0)
    if pos_weight is None:
        pos_weight = 3.0

    return EvalConfig(
        seed=project_seed,
        device=str(training_cfg.get("device", "cuda")),
        batch_size=int(training_cfg.get("batch_size", 128)),
        n_splits=int(split_cfg.get("n_splits", 5)),
        random_state=int(split_cfg.get("random_state", 42)),
        positive_label=int(deep_get(data_cfg, ["labels_to_use", "positive"], 1)),
        negative_label=int(deep_get(data_cfg, ["labels_to_use", "negative"], 0)),
        ignore_label_value=int(data_cfg.get("ignore_label_value", -1)),
        drop_ignore_label=bool(data_cfg.get("drop_ignore_label", True)),
        positive_class_weight=float(pos_weight),
        threshold=validate_threshold(float(eval_cfg.get("threshold", 0.7910))),
        save_internal_predictions_csv=bool(eval_cfg.get("save_predictions_csv", True)),
        save_external_predictions_csv=bool(external_cfg.get("save_external_predictions", True)),
        save_fold_metrics_csv=bool(eval_cfg.get("save_fold_metrics_csv", True)),
        save_horizon_summary_csv=bool(deep_get(cfg_dict, ["comparison", "save_horizon_summary_table"], True)),
        save_confusion_matrix_csv=bool(eval_cfg.get("save_confusion_matrix", True)),
        save_main_performance_table_csv=bool(deep_get(cfg_dict, ["comparison", "save_horizon_summary_table"], True)),
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


def load_merged_internal_arrays(root_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal evaluation mirrors train.py:
    merge any available train/val/test arrays, then reconstruct patient-level folds.
    """
    parts = []
    for prefix in ("train", "val", "test"):
        triplet = _load_triplet(root_dir, prefix)
        if triplet is not None:
            parts.append(triplet)

    if not parts:
        raise FileNotFoundError(f"No valid train/val/test npy triplets found under: {root_dir}")

    x_all = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
    y_all = np.concatenate([p[1] for p in parts], axis=0).astype(np.int64)
    pid_all = np.concatenate([p[2] for p in parts], axis=0)
    return x_all, y_all, pid_all


def load_external_test_arrays(root_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    External evaluation is strict: only test_* arrays are used.
    """
    triplet = _load_triplet(root_dir, "test")
    if triplet is None:
        raise FileNotFoundError(
            f"External evaluation requires test_data.npy / test_label.npy / test_pid.npy under: {root_dir}"
        )
    return triplet


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
# Model / loader helpers
# =============================================================================

def build_model(input_dim: int, output_dim: int = 2, model_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    model_cfg = model_cfg or {}
    name = str(model_cfg.get("name", "resnet1d")).lower()
    dropout = float(model_cfg.get("dropout", 0.2))

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


def build_loss(device: torch.device, positive_class_weight: float) -> nn.Module:
    class_weights = torch.tensor([1.0, float(positive_class_weight)], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
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
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# =============================================================================
# Metrics
# =============================================================================

def safe_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, prob_pos))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, prob_pos))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auc": safe_auc(y_true, prob_pos),
        "auprc": safe_auprc(y_true, prob_pos),
    }


def summarize_metrics_dataframe(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["loss", "accuracy", "recall", "precision", "f1", "f2", "macro_f1", "auc", "auprc"]:
        if metric not in fold_df.columns:
            continue
        rows.append(
            {
                "metric": metric,
                "mean": float(fold_df[metric].mean()),
                "std": float(fold_df[metric].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    save_dataframe(df, path)


# =============================================================================
# Prediction / evaluation
# =============================================================================

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
def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
    positive_class_weight: float,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    threshold: float,
) -> Dict[str, Any]:
    input_dim = int(x_eval.shape[-1])
    model = build_model(input_dim=input_dim, output_dim=2, model_cfg=model_cfg).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    criterion = build_loss(device=device, positive_class_weight=positive_class_weight)
    loader = build_loader(
        x_eval,
        y_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
    )

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


def predictions_to_dataframe(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_pos: np.ndarray,
    pid: np.ndarray,
    split_name: str,
    dataset_name: str,
    horizon_name: str,
    horizon_minutes: int,
    fold: Optional[int],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "y_prob": prob_pos.astype(float),
            "pid": pid,
            "split": split_name,
            "dataset": dataset_name,
            "horizon": horizon_minutes,
            "horizon_name": horizon_name,
        }
    )
    if fold is not None:
        df["fold"] = int(fold)
    return df


# =============================================================================
# Internal evaluation
# =============================================================================

def evaluate_internal_cv(
    *,
    spec: HorizonSpec,
    cfg: EvalConfig,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    x_all, y_all, pid_all = load_merged_internal_arrays(spec.internal_dir)
    x_all, y_all, pid_all = filter_labels(
        x_all,
        y_all,
        pid_all,
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
        drop_ignore_label=cfg.drop_ignore_label,
        ignore_label_value=cfg.ignore_label_value,
    )

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    fold_rows: List[Dict[str, Any]] = []
    pred_parts: List[pd.DataFrame] = []

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(x_all, y_all, groups=pid_all), start=1):
        ckpt_path = spec.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {ckpt_path}")

        x_val = x_all[val_idx]
        y_val = y_all[val_idx]
        pid_val = pid_all[val_idx]

        eval_res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            x_eval=x_val,
            y_eval=y_val,
            batch_size=cfg.batch_size,
            positive_class_weight=cfg.positive_class_weight,
            model_cfg=model_cfg,
            device=device,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            threshold=cfg.threshold,
        )

        metrics = eval_res["metrics"]
        pred_bundle = eval_res["pred_bundle"]

        fold_rows.append(
            {
                "fold": fold_idx,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "macro_f1": metrics["macro_f1"],
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "n_val_samples": int(len(y_val)),
                "n_val_patients": int(len(np.unique(pid_val))),
                "checkpoint_path": str(ckpt_path),
            }
        )

        if cfg.save_internal_predictions_csv:
            pred_parts.append(
                predictions_to_dataframe(
                    y_true=pred_bundle["trues"],
                    y_pred=pred_bundle["preds"],
                    prob_pos=pred_bundle["probs"][:, 1],
                    pid=pid_val,
                    split_name="internal_val",
                    dataset_name="MIMIC-III",
                    horizon_name=spec.name,
                    horizon_minutes=spec.minutes,
                    fold=fold_idx,
                )
            )

        if cfg.save_confusion_matrix_csv:
            save_confusion_matrix_csv(
                metrics["confusion_matrix"],
                spec.output_dir / f"eval_fold_{fold_idx}_confusion_matrix.csv",
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_metrics_dataframe(fold_df)

    if cfg.save_fold_metrics_csv:
        save_dataframe(fold_df, spec.output_dir / "eval_internal_fold_metrics.csv")
    save_dataframe(summary_df, spec.output_dir / "eval_internal_mean_std_summary.csv")

    if pred_parts and cfg.save_internal_predictions_csv:
        pred_df = pd.concat(pred_parts, ignore_index=True)
        save_dataframe(pred_df, spec.output_dir / "eval_internal_predictions.csv")
    else:
        pred_df = pd.DataFrame()

    return {
        "fold_df": fold_df,
        "summary_df": summary_df,
        "pred_df": pred_df,
        "n_samples": int(len(y_all)),
        "n_patients": int(len(np.unique(pid_all))),
    }


# =============================================================================
# External evaluation: five-fold summary
# =============================================================================

def evaluate_external_fivefold(
    *,
    spec: HorizonSpec,
    cfg: EvalConfig,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if spec.external_dir is None or not spec.external_dir.exists():
        return None

    x_test, y_test, pid_test = load_external_test_arrays(spec.external_dir)
    x_test, y_test, pid_test = filter_labels(
        x_test,
        y_test,
        pid_test,
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
        drop_ignore_label=cfg.drop_ignore_label,
        ignore_label_value=cfg.ignore_label_value,
    )

    fold_rows: List[Dict[str, Any]] = []
    pred_parts: List[pd.DataFrame] = []

    for fold_idx in range(1, cfg.n_splits + 1):
        ckpt_path = spec.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint for external evaluation: {ckpt_path}")

        eval_res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            x_eval=x_test,
            y_eval=y_test,
            batch_size=cfg.batch_size,
            positive_class_weight=cfg.positive_class_weight,
            model_cfg=model_cfg,
            device=device,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            threshold=cfg.threshold,
        )

        metrics = eval_res["metrics"]
        pred_bundle = eval_res["pred_bundle"]

        fold_rows.append(
            {
                "fold": fold_idx,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "macro_f1": metrics["macro_f1"],
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "n_test_samples": int(len(y_test)),
                "n_test_patients": int(len(np.unique(pid_test))),
                "checkpoint_path": str(ckpt_path),
            }
        )

        if cfg.save_external_predictions_csv:
            pred_parts.append(
                predictions_to_dataframe(
                    y_true=pred_bundle["trues"],
                    y_pred=pred_bundle["preds"],
                    prob_pos=pred_bundle["probs"][:, 1],
                    pid=pid_test,
                    split_name="external_test",
                    dataset_name="MC-MED",
                    horizon_name=spec.name,
                    horizon_minutes=spec.minutes,
                    fold=fold_idx,
                )
            )

        if cfg.save_confusion_matrix_csv:
            save_confusion_matrix_csv(
                metrics["confusion_matrix"],
                spec.output_dir / "external" / f"eval_external_fold_{fold_idx}_confusion_matrix.csv",
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_metrics_dataframe(fold_df)

    save_dataframe(fold_df, spec.output_dir / "external" / "eval_external_fold_metrics.csv")
    save_dataframe(summary_df, spec.output_dir / "external" / "eval_external_mean_std_summary.csv")

    if pred_parts and cfg.save_external_predictions_csv:
        pred_df = pd.concat(pred_parts, ignore_index=True)
        save_dataframe(pred_df, spec.output_dir / "external" / "eval_external_predictions.csv")
    else:
        pred_df = pd.DataFrame()

    return {
        "fold_df": fold_df,
        "summary_df": summary_df,
        "pred_df": pred_df,
        "n_samples": int(len(y_test)),
        "n_patients": int(len(np.unique(pid_test))),
    }


# =============================================================================
# Optional external single-model evaluation
# =============================================================================

def evaluate_external_single_model(
    *,
    spec: HorizonSpec,
    cfg: EvalConfig,
    model_cfg: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if spec.external_dir is None or not spec.external_dir.exists():
        return None

    ckpt_path = spec.output_dir / "external" / "external_best_model.pth"
    if not ckpt_path.exists():
        return None

    x_test, y_test, pid_test = load_external_test_arrays(spec.external_dir)
    x_test, y_test, pid_test = filter_labels(
        x_test,
        y_test,
        pid_test,
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
        drop_ignore_label=cfg.drop_ignore_label,
        ignore_label_value=cfg.ignore_label_value,
    )

    eval_res = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        x_eval=x_test,
        y_eval=y_test,
        batch_size=cfg.batch_size,
        positive_class_weight=cfg.positive_class_weight,
        model_cfg=model_cfg,
        device=device,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        threshold=cfg.threshold,
    )

    metrics = eval_res["metrics"]
    pred_bundle = eval_res["pred_bundle"]

    metrics_df = pd.DataFrame(
        [
            {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "macro_f1": metrics["macro_f1"],
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "n_test_samples": int(len(y_test)),
                "n_test_patients": int(len(np.unique(pid_test))),
                "checkpoint_path": str(ckpt_path),
            }
        ]
    )
    save_dataframe(metrics_df, spec.output_dir / "external" / "eval_external_single_metrics.csv")

    pred_df = predictions_to_dataframe(
        y_true=pred_bundle["trues"],
        y_pred=pred_bundle["preds"],
        prob_pos=pred_bundle["probs"][:, 1],
        pid=pid_test,
        split_name="external_test_single_model",
        dataset_name="MC-MED",
        horizon_name=spec.name,
        horizon_minutes=spec.minutes,
        fold=None,
    )
    save_dataframe(pred_df, spec.output_dir / "external" / "eval_external_single_predictions.csv")

    if cfg.save_confusion_matrix_csv:
        save_confusion_matrix_csv(
            metrics["confusion_matrix"],
            spec.output_dir / "external" / "eval_external_single_confusion_matrix.csv",
        )

    return {
        "metrics_df": metrics_df,
        "pred_df": pred_df,
        "n_samples": int(len(y_test)),
        "n_patients": int(len(np.unique(pid_test))),
    }


# =============================================================================
# Main performance table
# =============================================================================

def summary_df_to_metric_map(summary_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in summary_df.iterrows():
        out[str(row["metric"])] = (float(row["mean"]), float(row["std"]))
    return out


def build_main_performance_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for res in results:
        spec = res["spec"]

        # Internal row
        internal_map = summary_df_to_metric_map(res["internal"]["summary_df"])
        rows.append(
            {
                "Window": f"{spec.minutes} min",
                "Dataset": "MIMIC-III",
                "Accuracy": format_mean_std(*internal_map.get("accuracy", (np.nan, np.nan))),
                "Recall": format_mean_std(*internal_map.get("recall", (np.nan, np.nan))),
                "Precision": format_mean_std(*internal_map.get("precision", (np.nan, np.nan))),
                "F1-score": format_mean_std(*internal_map.get("f1", (np.nan, np.nan))),
                "F2-score": format_mean_std(*internal_map.get("f2", (np.nan, np.nan))),
                "AUC": format_mean_std(*internal_map.get("auc", (np.nan, np.nan))),
            }
        )

        # External row: five-fold summary over the 5 fold checkpoints
        external_fivefold = res.get("external_fivefold")
        if external_fivefold is not None:
            external_map = summary_df_to_metric_map(external_fivefold["summary_df"])
            rows.append(
                {
                    "Window": "",
                    "Dataset": "MC-MED",
                    "Accuracy": format_mean_std(*external_map.get("accuracy", (np.nan, np.nan))),
                    "Recall": format_mean_std(*external_map.get("recall", (np.nan, np.nan))),
                    "Precision": format_mean_std(*external_map.get("precision", (np.nan, np.nan))),
                    "F1-score": format_mean_std(*external_map.get("f1", (np.nan, np.nan))),
                    "F2-score": format_mean_std(*external_map.get("f2", (np.nan, np.nan))),
                    "AUC": format_mean_std(*external_map.get("auc", (np.nan, np.nan))),
                }
            )

    return pd.DataFrame(rows)


def summarize_horizons(results: List[Dict[str, Any]], output_root: Path) -> None:
    rows = []

    for res in results:
        spec = res["spec"]
        internal_summary = res["internal"]["summary_df"]
        ext_fivefold = res.get("external_fivefold")

        row = {
            "horizon_name": spec.name,
            "minutes": spec.minutes,
        }

        for _, r in internal_summary.iterrows():
            row[f"internal_{r['metric']}_mean"] = r["mean"]
            row[f"internal_{r['metric']}_std"] = r["std"]

        if ext_fivefold is not None:
            for _, r in ext_fivefold["summary_df"].iterrows():
                row[f"external_{r['metric']}_mean"] = r["mean"]
                row[f"external_{r['metric']}_std"] = r["std"]

        rows.append(row)

    df = pd.DataFrame(rows)
    save_dataframe(df, output_root / "eval_all_horizon_summary.csv")


# =============================================================================
# Main runner
# =============================================================================

def run_single_horizon_eval(
    *,
    spec: HorizonSpec,
    cfg: EvalConfig,
    full_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    device = to_device(cfg.device)
    model_cfg = deep_get(full_cfg, ["model"], {}) or {}

    internal_res = evaluate_internal_cv(
        spec=spec,
        cfg=cfg,
        model_cfg=model_cfg,
        device=device,
    )

    external_fivefold_res = evaluate_external_fivefold(
        spec=spec,
        cfg=cfg,
        model_cfg=model_cfg,
        device=device,
    )

    external_single_res = evaluate_external_single_model(
        spec=spec,
        cfg=cfg,
        model_cfg=model_cfg,
        device=device,
    )

    manifest = {
        "horizon_name": spec.name,
        "minutes": spec.minutes,
        "internal_dir": str(spec.internal_dir),
        "external_dir": str(spec.external_dir) if spec.external_dir is not None else None,
        "output_dir": str(spec.output_dir),
        "device": str(device),
        "internal_n_samples": internal_res["n_samples"],
        "internal_n_patients": internal_res["n_patients"],
        "external_n_samples": external_fivefold_res["n_samples"] if external_fivefold_res is not None else None,
        "external_n_patients": external_fivefold_res["n_patients"] if external_fivefold_res is not None else None,
        "external_eval_mode": "fivefold_checkpoints_on_test" if external_fivefold_res is not None else None,
        "f2_enabled": True,
        "operating_threshold": cfg.threshold,
    }
    save_json(manifest, spec.output_dir / "eval_manifest.json")

    return {
        "spec": spec,
        "manifest": manifest,
        "internal": internal_res,
        "external_fivefold": external_fivefold_res,
        "external_single": external_single_res,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints for the configured classifier.")
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
    eval_cfg = build_eval_config(cfg_dict)

    if args.single_internal_dir is not None:
        spec = HorizonSpec(
            name=str(args.single_name),
            minutes=int(args.single_minutes),
            internal_dir=Path(args.single_internal_dir),
            external_dir=Path(args.single_external_dir) if args.single_external_dir else None,
            output_dir=Path(args.single_output_dir) if args.single_output_dir else Path("outputs") / str(args.single_name),
        )
        results = [run_single_horizon_eval(spec=spec, cfg=eval_cfg, full_cfg=cfg_dict)]
        summarize_horizons(results, spec.output_dir.parent)

        main_table = build_main_performance_table(results)
        save_dataframe(main_table, spec.output_dir.parent / "eval_main_performance_table.csv")

        print(f"Finished evaluation: {spec.name}")
        return

    specs = build_horizon_specs(cfg_dict)
    results = []

    for spec in specs:
        print("=" * 88)
        print(f"Evaluating horizon: {spec.name} ({spec.minutes} min)")
        print(f"Internal dir: {spec.internal_dir}")
        print(f"External dir: {spec.external_dir}")
        print(f"Output dir  : {spec.output_dir}")
        res = run_single_horizon_eval(spec=spec, cfg=eval_cfg, full_cfg=cfg_dict)
        results.append(res)

    common_root = Path("outputs") / "main"
    ensure_dir(common_root)

    summarize_horizons(results, common_root)
    main_table = build_main_performance_table(results)
    save_dataframe(main_table, common_root / "eval_main_performance_table.csv")

    print("=" * 88)
    print("Evaluation finished.")
    print(f"Horizon summary: {common_root / 'eval_all_horizon_summary.csv'}")
    print(f"Main table     : {common_root / 'eval_main_performance_table.csv'}")


if __name__ == "__main__":
    main()
