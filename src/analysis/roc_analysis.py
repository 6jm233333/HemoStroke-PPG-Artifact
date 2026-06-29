
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


_TRUE_ALIASES = ["y_true", "label", "target", "gt", "true_label"]
_SCORE_ALIASES = ["y_prob", "prob", "score", "pred_prob", "positive_prob", "prob_1", "p1"]
_DATASET_ALIASES = ["dataset", "cohort", "source"]
_HORIZON_ALIASES = ["horizon", "window", "window_min", "lead_time_min", "minutes", "horizon_min"]
_FOLD_ALIASES = ["fold", "cv_fold", "split", "fold_id"]


def _find_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = {str(c).strip(): c for c in columns}
    for cand in candidates:
        if cand in colset:
            return colset[cand]
    return None


def _resolve_column(df: pd.DataFrame, explicit: Optional[str], aliases: Sequence[str], required: bool = True) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Column '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit
    col = _find_first_existing(df.columns, aliases)
    if required and col is None:
        raise KeyError(f"Could not resolve column from aliases={aliases}. Available columns: {list(df.columns)}")
    return col


def _validate_binary_labels(y_true: np.ndarray, pos_label: int = 1) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    uniques = np.unique(y_true)
    if len(uniques) != 2:
        raise ValueError(f"ROC/AUC requires binary labels, got unique values: {uniques.tolist()}")
    return (y_true == pos_label).astype(int)


def compute_roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: int = 1,
    drop_intermediate: bool = False,
) -> pd.DataFrame:
    """
    Compute ROC curve points for one binary prediction set.

    Returns a DataFrame with:
    - fpr
    - tpr
    - thresholds
    """
    y_true_arr = _validate_binary_labels(np.asarray(y_true), pos_label=pos_label)
    y_score_arr = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(
        y_true_arr,
        y_score_arr,
        pos_label=1,
        drop_intermediate=drop_intermediate,
    )
    return pd.DataFrame(
        {
            "fpr": fpr.astype(float),
            "tpr": tpr.astype(float),
            "threshold": thresholds.astype(float),
        }
    )


def compute_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: int = 1,
) -> float:
    """
    Compute ROC-AUC for one binary prediction set.
    """
    y_true_arr = _validate_binary_labels(np.asarray(y_true), pos_label=pos_label)
    y_score_arr = np.asarray(y_score, dtype=float)
    return float(roc_auc_score(y_true_arr, y_score_arr))


def _interpolate_roc(roc_df: pd.DataFrame, fpr_grid: np.ndarray) -> np.ndarray:
    x = roc_df["fpr"].to_numpy(dtype=float)
    y = roc_df["tpr"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    interp = np.interp(fpr_grid, x_unique, y_unique)
    interp[0] = 0.0
    interp[-1] = 1.0
    return interp


def summarize_multi_horizon_roc(
    pred_df: pd.DataFrame,
    *,
    y_true_col: Optional[str] = None,
    y_score_col: Optional[str] = None,
    dataset_col: Optional[str] = None,
    horizon_col: Optional[str] = None,
    fold_col: Optional[str] = None,
    pos_label: int = 1,
    fpr_grid: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate ROC across (dataset, horizon, fold).

    Expected minimum columns:
    - y_true
    - y_prob / score

    Optional:
    - dataset
    - horizon
    - fold

    Returns a dict:
    - summary: dataset/horizon level AUC summary
    - roc_points: interpolated mean/std ROC points
    - per_fold_roc: raw per-fold ROC points
    - per_fold_auc: per-fold AUC table
    """
    if pred_df.empty:
        raise ValueError("pred_df is empty.")

    df = pred_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    y_true_col = _resolve_column(df, y_true_col, _TRUE_ALIASES, required=True)
    y_score_col = _resolve_column(df, y_score_col, _SCORE_ALIASES, required=True)
    dataset_col = _resolve_column(df, dataset_col, _DATASET_ALIASES, required=False)
    horizon_col = _resolve_column(df, horizon_col, _HORIZON_ALIASES, required=False)
    fold_col = _resolve_column(df, fold_col, _FOLD_ALIASES, required=False)

    if dataset_col is None:
        df["dataset"] = "dataset"
        dataset_col = "dataset"
    if horizon_col is None:
        df["horizon"] = "all"
        horizon_col = "horizon"
    if fold_col is None:
        df["fold"] = 0
        fold_col = "fold"

    df = df[[dataset_col, horizon_col, fold_col, y_true_col, y_score_col]].copy()
    df = df.dropna(subset=[y_true_col, y_score_col])

    if fpr_grid is None:
        fpr_grid = np.linspace(0.0, 1.0, 201)

    summary_rows = []
    per_fold_auc_rows = []
    per_fold_roc_parts = []
    roc_point_rows = []

    group_keys = [dataset_col, horizon_col]
    for (dataset_name, horizon_value), group_df in df.groupby(group_keys, dropna=False, sort=True):
        aucs = []
        interp_tprs = []

        for fold_value, fold_df in group_df.groupby(fold_col, dropna=False, sort=True):
            y_true = fold_df[y_true_col].to_numpy()
            y_score = fold_df[y_score_col].to_numpy()

            roc_df = compute_roc_curve(y_true, y_score, pos_label=pos_label)
            auc_val = compute_auc(y_true, y_score, pos_label=pos_label)

            fold_roc = roc_df.copy()
            fold_roc["dataset"] = dataset_name
            fold_roc["horizon"] = horizon_value
            fold_roc["fold"] = fold_value
            per_fold_roc_parts.append(fold_roc)

            aucs.append(auc_val)
            interp_tprs.append(_interpolate_roc(roc_df, fpr_grid))

            per_fold_auc_rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon_value,
                    "fold": fold_value,
                    "auc": auc_val,
                    "n_samples": int(len(fold_df)),
                    "positive_rate": float((fold_df[y_true_col] == pos_label).mean()),
                }
            )

        interp_arr = np.vstack(interp_tprs)
        mean_tpr = interp_arr.mean(axis=0)
        std_tpr = interp_arr.std(axis=0, ddof=0)

        for fpr_val, tpr_mean_val, tpr_std_val in zip(fpr_grid, mean_tpr, std_tpr):
            roc_point_rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon_value,
                    "fpr": float(fpr_val),
                    "mean_tpr": float(tpr_mean_val),
                    "std_tpr": float(tpr_std_val),
                    "n_folds": int(len(aucs)),
                }
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "horizon": horizon_value,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=0)),
                "auc_sem": float(np.std(aucs, ddof=0) / np.sqrt(max(len(aucs), 1))),
                "n_folds": int(len(aucs)),
                "n_rows": int(len(group_df)),
                "positive_rate": float((group_df[y_true_col] == pos_label).mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    per_fold_auc_df = pd.DataFrame(per_fold_auc_rows)
    roc_points_df = pd.DataFrame(roc_point_rows)
    per_fold_roc_df = pd.concat(per_fold_roc_parts, ignore_index=True) if per_fold_roc_parts else pd.DataFrame()

    if not summary_df.empty:
        def _sort_horizon_key(v):
            try:
                return float(v)
            except Exception:
                return str(v)

        summary_df = summary_df.sort_values(by=["dataset", "horizon"], key=lambda s: s.map(_sort_horizon_key)).reset_index(drop=True)
        per_fold_auc_df = per_fold_auc_df.sort_values(by=["dataset", "horizon", "fold"]).reset_index(drop=True)
        roc_points_df = roc_points_df.sort_values(by=["dataset", "horizon", "fpr"]).reset_index(drop=True)
        if not per_fold_roc_df.empty:
            per_fold_roc_df = per_fold_roc_df.sort_values(by=["dataset", "horizon", "fold", "fpr"]).reset_index(drop=True)

    return {
        "summary": summary_df,
        "roc_points": roc_points_df,
        "per_fold_roc": per_fold_roc_df,
        "per_fold_auc": per_fold_auc_df,
    }


def export_roc_points(
    roc_result: Union[Mapping[str, pd.DataFrame], pd.DataFrame],
    output_path: Union[str, Path],
    *,
    table_key: str = "roc_points",
    index: bool = False,
) -> Path:
    """
    Export ROC points table to CSV.

    roc_result:
    - either the dict returned by summarize_multi_horizon_roc
    - or a DataFrame directly
    """
    if isinstance(roc_result, pd.DataFrame):
        df = roc_result
    else:
        if table_key not in roc_result:
            raise KeyError(f"table_key='{table_key}' not in roc_result. Available keys: {list(roc_result.keys())}")
        df = roc_result[table_key]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path
