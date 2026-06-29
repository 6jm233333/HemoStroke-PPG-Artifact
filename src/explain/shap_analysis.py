
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.ndimage import gaussian_filter1d


SHAP_CMAP = plt.get_cmap("coolwarm")


def set_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def save_dual_formats(outname: str | Path) -> tuple[Path, Path]:
    outname = Path(outname)
    outname.parent.mkdir(parents=True, exist_ok=True)
    png_path = outname.with_suffix(".png")
    pdf_path = outname.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return png_path, pdf_path


def ensure_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def load_feature_names(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data]
    if path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError("feature-names file must be .json or .txt")


def squeeze_binary_class_dimension(arr: np.ndarray) -> np.ndarray:
    arr = ensure_numpy(arr)
    # common cases:
    # [N, T, F]
    # [N, F]
    # [N, T, F, 1]
    # [N, T, F, 2] -> choose positive class
    # [N, F, 2] -> choose positive class
    if arr.ndim >= 3 and arr.shape[-1] == 2:
        return arr[..., 1]
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_binary_class_dimension(arr)
    if arr.ndim == 2:
        # [N, F] -> [N, 1, F]
        return arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array after squeezing, got shape={arr.shape}")
    return arr


def aggregate_over_time(arr_3d: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return arr_3d.mean(axis=1)
    if mode == "sum":
        return arr_3d.sum(axis=1)
    if mode == "maxabs":
        idx = np.abs(arr_3d).argmax(axis=1)
        out = np.take_along_axis(arr_3d, idx[:, None, :], axis=1)[:, 0, :]
        return out
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def compute_global_importance(
    shap_values_3d: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    abs_per_sample = np.mean(np.abs(shap_values_3d), axis=1)   # [N, F]
    mean_abs = abs_per_sample.mean(axis=0)
    sem_abs = abs_per_sample.std(axis=0, ddof=1) / max(np.sqrt(len(abs_per_sample)), 1.0)

    df = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": mean_abs,
        "sem_abs_shap": sem_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def plot_global_bar(
    importance_df: pd.DataFrame,
    title_suffix: str = "",
    outname: str | Path = "global_shap_importance",
    top_k: int = 20,
    err_label: str = "SEM across samples",
    title_main: str = "Global feature importance",
) -> None:
    set_publication_style()
    df = importance_df.head(top_k).iloc[::-1].copy()

    plt.figure(figsize=(8, max(4.8, 0.36 * len(df) + 1.2)))
    values = df["mean_abs_shap"].to_numpy()
    xerr = df["sem_abs_shap"].to_numpy()
    names = df["feature"].tolist()
    y = np.arange(len(df))

    plt.barh(
        y,
        values,
        xerr=xerr,
        color="#4C78A8",
        alpha=0.92,
        height=0.72,
        error_kw={"elinewidth": 1.0, "capsize": 2, "ecolor": "black"},
    )

    plt.yticks(y, names, fontsize=11)
    plt.xlabel(r"Mean $|$SHAP value$|$ (impact magnitude)", fontsize=12)

    suffix_clean = " ".join(str(title_suffix).replace("_", " ").split())
    title = title_main if suffix_clean == "" else f"{title_main} ({suffix_clean})"
    plt.title(title, fontsize=13, pad=12)

    xmax = float(np.max(values + xerr)) if len(values) else 1.0
    plt.xlim(0, xmax * 1.08)
    plt.margins(x=0.02)

    plt.text(
        0.99,
        0.02,
        f"Error bar: {err_label}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    sns.despine(top=True, right=True)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_dual_formats(outname)
    plt.close()


def plot_summary_beeswarm(
    shap_vals_signed_2d: np.ndarray,
    features_2d: np.ndarray,
    feature_names: Sequence[str],
    title_suffix: str = "",
    outname: str | Path = "shap_beeswarm",
    max_display: int | None = None,
) -> None:
    set_publication_style()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_vals_signed_2d,
        features_2d,
        feature_names=list(feature_names),
        show=False,
        cmap=SHAP_CMAP,
        max_display=max_display or len(feature_names),
        plot_size=(8, 6),
    )
    ax = plt.gca()
    suffix_clean = " ".join(str(title_suffix).replace("_", " ").split())
    title = "Feature impact directionality" if suffix_clean == "" else f"Feature impact directionality ({suffix_clean})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=12)
    plt.tight_layout()
    save_dual_formats(outname)
    plt.close()


def plot_local_heatmap(
    shap_val_single_2d: np.ndarray,
    feature_names: Sequence[str],
    sample_id: str | int,
    prob: float | None = None,
    title_suffix: str = "",
    outname: str | Path = "shap_local_heatmap",
    smooth_sigma: float = 5.0,
) -> None:
    set_publication_style()
    heat = ensure_numpy(shap_val_single_2d).T
    heat_smooth = gaussian_filter1d(heat, sigma=smooth_sigma, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    limit = np.percentile(np.abs(heat_smooth), 99.5)
    limit = float(limit) if np.isfinite(limit) and limit > 0 else 1.0

    im = ax.imshow(
        heat_smooth,
        aspect="auto",
        cmap=SHAP_CMAP,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel(r"Time step ($t$)", fontsize=12)

    suffix_clean = " ".join(str(title_suffix).replace("_", " ").split())
    if prob is None:
        title = f"Temporal importance map (ID: {sample_id})"
    else:
        title = f"Temporal importance map (ID: {sample_id}, P={prob:.2f})"
    if suffix_clean:
        title += f" ({suffix_clean})"
    ax.set_title(title, fontsize=13)

    cbar = plt.colorbar(im, pad=0.02, aspect=30)
    cbar.set_label("SHAP value", rotation=270, labelpad=15, fontsize=10)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout()
    save_dual_formats(outname)
    plt.close()


def plot_dependence_grid_topk(
    shap_vals_signed_2d: np.ndarray,
    features_2d: np.ndarray,
    feature_names: Sequence[str],
    importance_df: pd.DataFrame,
    title_suffix: str = "",
    outname: str | Path = "shap_dependence_grid",
    top_k: int = 9,
    ncols: int = 3,
    point_size: int = 18,
    alpha: float = 0.55,
) -> None:
    set_publication_style()

    top_features = importance_df["feature"].head(top_k).tolist()
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    top_features = [f for f in top_features if f in feature_to_idx]
    if not top_features:
        return

    nrows = int(np.ceil(len(top_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[len(top_features):]:
        ax.axis("off")

    for ax, feat in zip(axes_flat, top_features):
        idx = feature_to_idx[feat]
        x = features_2d[:, idx]
        y = shap_vals_signed_2d[:, idx]

        ax.scatter(x, y, s=point_size, alpha=alpha, color="#4C78A8", edgecolors="none")
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("Feature value", fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.25)

    suffix_clean = " ".join(str(title_suffix).replace("_", " ").split())
    title = "SHAP dependence plots" if suffix_clean == "" else f"SHAP dependence plots ({suffix_clean})"
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    save_dual_formats(outname)
    plt.close(fig)


def save_global_tables(importance_df: pd.DataFrame, out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_csv, index=False)
    return out_csv


def prepare_summary_inputs(
    shap_values_3d: np.ndarray,
    features_3d: np.ndarray,
    aggregate_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    shap_2d = aggregate_over_time(shap_values_3d, mode=aggregate_mode)
    feat_2d = aggregate_over_time(features_3d, mode=aggregate_mode)
    return shap_2d, feat_2d


def compute_shap_values_torch(
    model: Any,
    background: np.ndarray,
    evaluation: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generic PyTorch wrapper for sequence models.
    Returns SHAP values with shape [N, T, F] or [N, F].
    """
    import torch

    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model: Any):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base_model(x)
            if out.ndim == 2 and out.shape[1] == 2:
                return torch.softmax(out, dim=1)[:, 1:2]
            if out.ndim == 2 and out.shape[1] == 1:
                return torch.sigmoid(out)
            if out.ndim == 1:
                return out[:, None]
            return out

    model = model.to(device)
    model.eval()
    wrapped = WrappedModel(model).to(device)

    bg = torch.tensor(background, dtype=torch.float32, device=device)
    ev = torch.tensor(evaluation, dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapped, bg)

    out_chunks = []
    for start in range(0, len(ev), batch_size):
        end = min(start + batch_size, len(ev))
        chunk = ev[start:end]
        shap_chunk = explainer.shap_values(chunk)
        if isinstance(shap_chunk, list):
            shap_chunk = shap_chunk[-1]
        out_chunks.append(ensure_numpy(shap_chunk))

    shap_values = np.concatenate(out_chunks, axis=0)
    return squeeze_binary_class_dimension(shap_values)


def build_full_shap_report(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    output_dir: str | Path,
    sample_ids: Sequence[Any] | None = None,
    probs: Sequence[float] | None = None,
    title_suffix: str = "",
    top_k_bar: int = 20,
    top_k_dependence: int = 9,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_3d = ensure_3d(shap_values)
    feat_3d = ensure_3d(features)

    if shap_3d.shape != feat_3d.shape:
        raise ValueError(f"SHAP and feature arrays must have identical shape after normalization. "
                         f"Got SHAP={shap_3d.shape}, features={feat_3d.shape}")

    if shap_3d.shape[-1] != len(feature_names):
        raise ValueError(
            f"feature_names length mismatch: got {len(feature_names)}, expected {shap_3d.shape[-1]}"
        )

    importance_df = compute_global_importance(shap_3d, feature_names)
    importance_csv = save_global_tables(importance_df, output_dir / "global_feature_importance.csv")

    plot_global_bar(
        importance_df=importance_df,
        title_suffix=title_suffix,
        outname=output_dir / "fig1_global_importance",
        top_k=top_k_bar,
    )

    shap_2d, feat_2d = prepare_summary_inputs(shap_3d, feat_3d, aggregate_mode="mean")

    plot_summary_beeswarm(
        shap_vals_signed_2d=shap_2d,
        features_2d=feat_2d,
        feature_names=feature_names,
        title_suffix=title_suffix,
        outname=output_dir / "fig2_beeswarm",
        max_display=min(len(feature_names), top_k_bar),
    )

    if shap_3d.shape[1] > 1:
        local_idx = int(np.argmax(np.mean(np.abs(shap_3d), axis=(1, 2))))
        sample_id = sample_ids[local_idx] if sample_ids is not None else local_idx
        prob = float(probs[local_idx]) if probs is not None else None

        plot_local_heatmap(
            shap_val_single_2d=shap_3d[local_idx],
            feature_names=feature_names,
            sample_id=sample_id,
            prob=prob,
            title_suffix=title_suffix,
            outname=output_dir / "fig3_local_heatmap",
        )

    plot_dependence_grid_topk(
        shap_vals_signed_2d=shap_2d,
        features_2d=feat_2d,
        feature_names=feature_names,
        importance_df=importance_df,
        title_suffix=title_suffix,
        outname=output_dir / "fig4_dependence_grid",
        top_k=top_k_dependence,
    )

    manifest = {
        "global_importance_csv": str(importance_csv),
        "output_dir": str(output_dir),
        "n_samples": int(shap_3d.shape[0]),
        "time_steps": int(shap_3d.shape[1]),
        "n_features": int(shap_3d.shape[2]),
    }
    manifest_path = output_dir / "shap_report_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "global_importance_csv": str(importance_csv),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a publication-quality SHAP report from precomputed SHAP arrays.")
    parser.add_argument("--shap-values-npy", type=str, required=True, help="Path to shap_values.npy")
    parser.add_argument("--features-npy", type=str, required=True, help="Path to features.npy aligned with SHAP values")
    parser.add_argument("--feature-names", type=str, required=True, help="Path to feature names .json or .txt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-ids-npy", type=str, default=None)
    parser.add_argument("--probs-npy", type=str, default=None)
    parser.add_argument("--title-suffix", type=str, default="")
    args = parser.parse_args()

    shap_values = np.load(args.shap_values_npy, allow_pickle=True)
    features = np.load(args.features_npy, allow_pickle=True)
    feature_names = load_feature_names(args.feature_names)

    sample_ids = None
    probs = None
    if args.sample_ids_npy:
        sample_ids = np.load(args.sample_ids_npy, allow_pickle=True)
    if args.probs_npy:
        probs = np.load(args.probs_npy, allow_pickle=True)

    outputs = build_full_shap_report(
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        output_dir=args.output_dir,
        sample_ids=sample_ids,
        probs=probs,
        title_suffix=args.title_suffix,
    )

    print(f"[shap_analysis] global importance csv -> {outputs['global_importance_csv']}")
    print(f"[shap_analysis] manifest -> {outputs['manifest']}")


if __name__ == "__main__":
    main()
