from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for SHAP figure reproduction.")
    parser.add_argument("--shap-values-npy", required=True)
    parser.add_argument("--features-npy", required=True)
    parser.add_argument("--feature-names", default="configs/feature_set_17.json")
    parser.add_argument("--output-dir", default="outputs/figures/shap")
    parser.add_argument("--sample-ids-npy", default=None)
    parser.add_argument("--probs-npy", default=None)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.explain.shap_analysis",
        "--shap-values-npy",
        args.shap_values_npy,
        "--features-npy",
        args.features_npy,
        "--feature-names",
        args.feature_names,
        "--output-dir",
        args.output_dir,
        "--title-suffix",
        args.title_suffix,
    ]
    if args.sample_ids_npy:
        cmd.extend(["--sample-ids-npy", args.sample_ids_npy])
    if args.probs_npy:
        cmd.extend(["--probs-npy", args.probs_npy])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
