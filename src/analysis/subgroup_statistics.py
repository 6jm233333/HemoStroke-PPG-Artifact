
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


_PANEL_ALIASES = ["panel", "domain", "category", "subgroup_type", "panel_name"]
_SUBGROUP_ALIASES = ["subgroup", "group", "group_name", "level", "label"]
_FOLD_ALIASES = ["fold", "cv_fold", "split", "fold_id"]
_F1_ALIASES = ["f1", "f1_score", "macro_f1", "binary_f1"]


def _find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
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
    col = _find_first_existing(list(df.columns), aliases)
    if required and col is None:
        raise KeyError(f"Could not resolve column from aliases={aliases}. Available columns: {list(df.columns)}")
    return col


def significance_label(p: float) -> str:
    if pd.isna(p):
        return "ns"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return "ns"


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def adjust_pvalues(pvalues: Sequence[float], method: str = "bonferroni") -> np.ndarray:
    """
    Supported:
    - bonferroni
    - holm
    - fdr_bh
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    valid_mask = np.isfinite(p)
    pv = p[valid_mask]
    m = len(pv)

    if m == 0:
        return out

    method = method.lower()

    if method == "bonferroni":
        adj = np.minimum(pv * m, 1.0)

    elif method == "holm":
        order = np.argsort(pv)
        ranked = pv[order]
        holm = np.empty_like(ranked)
        for i, val in enumerate(ranked):
            holm[i] = min((m - i) * val, 1.0)
        holm = np.maximum.accumulate(holm)
        adj = np.empty_like(holm)
        adj[order] = holm

    elif method == "fdr_bh":
        order = np.argsort(pv)
        ranked = pv[order]
        bh = np.empty_like(ranked)
        for i, val in enumerate(ranked, start=1):
            bh[i - 1] = val * m / i
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0.0, 1.0)
        adj = np.empty_like(bh)
        adj[order] = bh

    else:
        raise ValueError(f"Unsupported p-value adjustment method: {method}")

    out[valid_mask] = adj
    return out


def summarize_subgroup_f1(
    df: pd.DataFrame,
    *,
    panel_col: Optional[str] = None,
    subgroup_col: Optional[str] = None,
    fold_col: Optional[str] = None,
    f1_col: Optional[str] = None,
    panel_order: Optional[Sequence[str]] = None,
    subgroup_order_map: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Summarize fold-level F1 into mean/std/sem for each subgroup.
    """
    if df.empty:
        raise ValueError("Input subgroup DataFrame is empty.")

    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    panel_col = _resolve_column(data, panel_col, _PANEL_ALIASES, required=False)
    subgroup_col = _resolve_column(data, subgroup_col, _SUBGROUP_ALIASES, required=True)
    fold_col = _resolve_column(data, fold_col, _FOLD_ALIASES, required=False)
    f1_col = _resolve_column(data, f1_col, _F1_ALIASES, required=True)

    if panel_col is None:
        data["panel"] = "panel"
        panel_col = "panel"
    if fold_col is None:
        data["fold"] = np.arange(len(data))
        fold_col = "fold"

    data[f1_col] = pd.to_numeric(data[f1_col], errors="coerce")
    data = data.dropna(subset=[f1_col])

    rows = []
    for (panel_name, subgroup_name), g in data.groupby([panel_col, subgroup_col], dropna=False, sort=False):
        values = g[f1_col].to_numpy(dtype=float)
        rows.append(
            {
                "panel": panel_name,
                "subgroup": subgroup_name,
                "n_folds": int(g[fold_col].nunique()),
                "n_rows": int(len(g)),
                "f1_mean": float(np.mean(values)),
                "f1_std": float(np.std(values, ddof=0)),
                "f1_sem": float(np.std(values, ddof=0) / np.sqrt(max(len(values), 1))),
                "f1_min": float(np.min(values)),
                "f1_max": float(np.max(values)),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    def _panel_rank(x: str) -> int:
        if panel_order is None:
            return 0
        try:
            return list(panel_order).index(x)
        except ValueError:
            return len(panel_order)

    def _subgroup_rank(panel_name: str, subgroup_name: str) -> int:
        if subgroup_order_map is None:
            return 0
        group_order = subgroup_order_map.get(panel_name)
        if group_order is None:
            return 0
        try:
            return list(group_order).index(subgroup_name)
        except ValueError:
            return len(group_order)

    out["_panel_rank"] = out["panel"].map(_panel_rank)
    out["_subgroup_rank"] = [
        _subgroup_rank(p, s) for p, s in zip(out["panel"], out["subgroup"])
    ]
    out = out.sort_values(["_panel_rank", "_subgroup_rank", "panel", "subgroup"]).drop(
        columns=["_panel_rank", "_subgroup_rank"]
    )
    out = out.reset_index(drop=True)
    return out


def run_two_group_test(
    df: pd.DataFrame,
    *,
    group_col: str = "subgroup",
    value_col: str = "f1",
    equal_var: bool = False,
) -> pd.DataFrame:
    """
    Welch t-test for exactly two groups.
    """
    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[group_col, value_col])

    groups = list(pd.unique(data[group_col]))
    if len(groups) != 2:
        raise ValueError(f"run_two_group_test expects exactly 2 groups, got {groups}")

    g1, g2 = groups
    x = data.loc[data[group_col] == g1, value_col].to_numpy(dtype=float)
    y = data.loc[data[group_col] == g2, value_col].to_numpy(dtype=float)

    stat, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
    return pd.DataFrame(
        [
            {
                "group_1": g1,
                "group_2": g2,
                "test": "welch_ttest" if not equal_var else "student_ttest",
                "statistic": float(stat),
                "p_raw": float(p),
                "p_adj": float(p),
                "p_value_label": significance_label(float(p)),
                "mean_1": float(np.mean(x)),
                "mean_2": float(np.mean(y)),
                "cohen_d": _cohen_d(x, y),
                "n_1": int(len(x)),
                "n_2": int(len(y)),
            }
        ]
    )


def run_multi_group_test(
    df: pd.DataFrame,
    *,
    group_col: str = "subgroup",
    value_col: str = "f1",
    correction: str = "bonferroni",
    equal_var: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    One-way ANOVA + pairwise Welch t-tests.
    """
    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[group_col, value_col])

    groups = list(pd.unique(data[group_col]))
    if len(groups) < 2:
        raise ValueError("run_multi_group_test requires at least 2 groups.")
    if len(groups) == 2:
        pairwise = run_two_group_test(data, group_col=group_col, value_col=value_col, equal_var=equal_var)
        global_df = pd.DataFrame(
            [
                {
                    "test": "two_group_only",
                    "statistic": np.nan,
                    "p_raw": pairwise.loc[0, "p_raw"],
                    "p_adj": pairwise.loc[0, "p_adj"],
                    "n_groups": 2,
                }
            ]
        )
        return {"global": global_df, "pairwise": pairwise}

    arrays = [data.loc[data[group_col] == g, value_col].to_numpy(dtype=float) for g in groups]
    f_stat, p_global = stats.f_oneway(*arrays)

    global_df = pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "statistic": float(f_stat),
                "p_raw": float(p_global),
                "p_adj": float(p_global),
                "n_groups": int(len(groups)),
            }
        ]
    )

    pair_rows = []
    for g1, g2 in combinations(groups, 2):
        x = data.loc[data[group_col] == g1, value_col].to_numpy(dtype=float)
        y = data.loc[data[group_col] == g2, value_col].to_numpy(dtype=float)
        stat, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
        pair_rows.append(
            {
                "group_1": g1,
                "group_2": g2,
                "test": "welch_ttest" if not equal_var else "student_ttest",
                "statistic": float(stat),
                "p_raw": float(p),
                "mean_1": float(np.mean(x)),
                "mean_2": float(np.mean(y)),
                "cohen_d": _cohen_d(x, y),
                "n_1": int(len(x)),
                "n_2": int(len(y)),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_adj"] = adjust_pvalues(pairwise["p_raw"].to_numpy(dtype=float), method=correction)
    pairwise["p_value_label"] = pairwise["p_adj"].map(significance_label)

    return {"global": global_df, "pairwise": pairwise}


def build_pairwise_pvalue_annotations(
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    *,
    panel: Optional[str] = None,
    subgroup_order: Optional[Sequence[str]] = None,
    mean_col: str = "f1_mean",
    err_col: str = "f1_std",
    alpha: float = 0.05,
    start_pad: float = 0.015,
    step: float = 0.03,
    text_pad: float = 0.006,
    only_significant: bool = True,
) -> List[Dict[str, float]]:
    """
    Build plotting annotations for pairwise p-value brackets.

    Returns a list of dicts:
    - x1
    - x2
    - y
    - text
    - p_adj
    - group_1
    - group_2
    """
    if summary_df.empty or pairwise_df.empty:
        return []

    s = summary_df.copy()
    p = pairwise_df.copy()

    if panel is not None and "panel" in s.columns:
        s = s[s["panel"] == panel].copy()
    if panel is not None and "panel" in p.columns:
        p = p[p["panel"] == panel].copy()

    if s.empty or p.empty:
        return []

    if subgroup_order is None:
        subgroup_order = s["subgroup"].tolist()

    x_map = {name: i for i, name in enumerate(subgroup_order)}

    s = s[s["subgroup"].isin(x_map)].copy()
    base_top = float((s[mean_col] + s[err_col]).max()) if not s.empty else 1.0

    if only_significant and "p_adj" in p.columns:
        p = p[p["p_adj"] < alpha].copy()
    if p.empty:
        return []

    p = p[p["group_1"].isin(x_map) & p["group_2"].isin(x_map)].copy()
    if p.empty:
        return []

    p["span"] = p.apply(lambda r: abs(x_map[r["group_2"]] - x_map[r["group_1"]]), axis=1)
    p = p.sort_values(["span", "p_adj"], ascending=[True, True]).reset_index(drop=True)

    annotations: List[Dict[str, float]] = []
    current_y = base_top + start_pad

    for _, row in p.iterrows():
        g1 = row["group_1"]
        g2 = row["group_2"]
        x1 = float(min(x_map[g1], x_map[g2]))
        x2 = float(max(x_map[g1], x_map[g2]))
        text = row["p_value_label"] if "p_value_label" in row else significance_label(float(row["p_adj"]))

        annotations.append(
            {
                "group_1": g1,
                "group_2": g2,
                "x1": x1,
                "x2": x2,
                "y": float(current_y),
                "text_y": float(current_y + text_pad),
                "text": text,
                "p_adj": float(row["p_adj"]) if "p_adj" in row else np.nan,
            }
        )
        current_y += step

    return annotations
