from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild temporal trajectory figures for one or more case CSVs.")
    parser.add_argument("--input", required=True, help="A case CSV or a directory containing case CSVs.")
    parser.add_argument("--output-dir", default="outputs/figures/temporal_cases")
    parser.add_argument("--feature-groups-json", default=None)
    parser.add_argument("--max-segments", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.rglob("*.csv"))
    else:
        files = [input_path]
    if not files:
        raise FileNotFoundError(f"No CSV files found for {input_path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for csv_path in files:
        cmd = [
            sys.executable,
            "-m",
            "src.explain.plot_feature_trajectories",
            "--input-csv",
            str(csv_path),
            "--output-dir",
            args.output_dir,
            "--max-segments",
            str(args.max_segments),
        ]
        if args.feature_groups_json:
            cmd.extend(["--feature-groups-json", args.feature_groups_json])
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
