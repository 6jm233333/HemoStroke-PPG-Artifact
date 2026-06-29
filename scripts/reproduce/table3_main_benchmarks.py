from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = ["Accuracy", "Precision", "Recall", "F1", "F2", "AUC"]


TABLE3_ROWS: list[dict[str, object]] = [
    {"Horizon": "240 min", "Cohort": "MIMIC-III", "Model": "Ours", "Accuracy": "0.6654 +/- 0.0047", "Precision": "0.6681 +/- 0.0041", "Recall": "0.9833 +/- 0.0106", "F1": "0.7956 +/- 0.0027", "F2": "0.8985 +/- 0.0062", "AUC": "0.6525 +/- 0.0530"},
    {"Horizon": "240 min", "Cohort": "MIMIC-III", "Model": "CHA2DS2-VASc", "Accuracy": "0.5550 +/- 0.0306", "Precision": "0.5890 +/- 0.0204", "Recall": "0.7440 +/- 0.0388", "F1": "0.6570 +/- 0.0250", "F2": "0.7068 +/- 0.0339", "AUC": "0.5061 +/- 0.0641"},
    {"Horizon": "240 min", "Cohort": "MIMIC-III", "Model": "Nwosu-EHR RF", "Accuracy": "0.5230 +/- 0.0352", "Precision": "0.5570 +/- 0.0378", "Recall": "0.8240 +/- 0.0344", "F1": "0.6650 +/- 0.0314", "F2": "0.7519 +/- 0.0367", "AUC": "0.6070 +/- 0.0940"},
    {"Horizon": "240 min", "Cohort": "MIMIC-III", "Model": "Teoh-EHR XGB", "Accuracy": "0.5730 +/- 0.0314", "Precision": "0.5730 +/- 0.0316", "Recall": "0.9920 +/- 0.0066", "F1": "0.7260 +/- 0.0268", "F2": "0.8654 +/- 0.0185", "AUC": "0.5655 +/- 0.0843"},
    {"Horizon": "240 min", "Cohort": "MIMIC-III", "Model": "Yang-EHR ML", "Accuracy": "0.5690 +/- 0.0327", "Precision": "0.5710 +/- 0.0337", "Recall": "0.9920 +/- 0.0066", "F1": "0.7250 +/- 0.0268", "F2": "0.8645 +/- 0.0196", "AUC": "0.5767 +/- 0.0716"},
    {"Horizon": "240 min", "Cohort": "MC-MED", "Model": "Ours", "Accuracy": "0.8636 +/- 0.0350", "Precision": "0.9179 +/- 0.0020", "Recall": "0.9341 +/- 0.0413", "F1": "0.9256 +/- 0.0211", "F2": "0.9306 +/- 0.0332", "AUC": "0.6195 +/- 0.0807"},
    {"Horizon": "240 min", "Cohort": "MC-MED", "Model": "CHA2DS2-VASc", "Accuracy": "0.6780 +/- 0.0469", "Precision": "0.8640 +/- 0.0191", "Recall": "0.7500 +/- 0.0503", "F1": "0.8030 +/- 0.0321", "F2": "0.7703 +/- 0.0456", "AUC": "0.4264 +/- 0.0771"},
    {"Horizon": "240 min", "Cohort": "MC-MED", "Model": "Nwosu-EHR RF", "Accuracy": "0.6780 +/- 0.0469", "Precision": "0.9000 +/- 0.0416", "Recall": "0.7110 +/- 0.0495", "F1": "0.7940 +/- 0.0385", "F2": "0.7422 +/- 0.0487", "AUC": "0.5370 +/- 0.0497"},
    {"Horizon": "240 min", "Cohort": "MC-MED", "Model": "Teoh-EHR XGB", "Accuracy": "0.6210 +/- 0.0500", "Precision": "0.9060 +/- 0.0390", "Recall": "0.6320 +/- 0.0559", "F1": "0.7440 +/- 0.0441", "F2": "0.6727 +/- 0.0550", "AUC": "0.2993 +/- 0.0530"},
    {"Horizon": "240 min", "Cohort": "MC-MED", "Model": "Yang-EHR ML", "Accuracy": "0.7240 +/- 0.0497", "Precision": "0.8610 +/- 0.0416", "Recall": "0.8160 +/- 0.0464", "F1": "0.8380 +/- 0.0339", "F2": "0.8246 +/- 0.0456", "AUC": "0.3098 +/- 0.0573"},
    {"Horizon": "300 min", "Cohort": "MIMIC-III", "Model": "Ours", "Accuracy": "0.7860 +/- 0.0129", "Precision": "0.8028 +/- 0.0125", "Recall": "0.9647 +/- 0.0360", "F1": "0.8759 +/- 0.0105", "F2": "0.9269 +/- 0.0243", "AUC": "0.6124 +/- 0.0668"},
    {"Horizon": "300 min", "Cohort": "MIMIC-III", "Model": "CHA2DS2-VASc", "Accuracy": "0.5990 +/- 0.0298", "Precision": "0.6910 +/- 0.0196", "Recall": "0.7420 +/- 0.0372", "F1": "0.7160 +/- 0.0235", "F2": "0.7312 +/- 0.0334", "AUC": "0.4999 +/- 0.1313"},
    {"Horizon": "300 min", "Cohort": "MIMIC-III", "Model": "Nwosu-EHR RF", "Accuracy": "0.6790 +/- 0.0324", "Precision": "0.6810 +/- 0.0298", "Recall": "0.9930 +/- 0.0054", "F1": "0.8080 +/- 0.0230", "F2": "0.9096 +/- 0.0143", "AUC": "0.5063 +/- 0.1252"},
    {"Horizon": "300 min", "Cohort": "MIMIC-III", "Model": "Teoh-EHR XGB", "Accuracy": "0.6500 +/- 0.0273", "Precision": "0.6780 +/- 0.0273", "Recall": "0.9930 +/- 0.0054", "F1": "0.8060 +/- 0.0212", "F2": "0.9086 +/- 0.0135", "AUC": "0.5656 +/- 0.0630"},
    {"Horizon": "300 min", "Cohort": "MIMIC-III", "Model": "Yang-EHR ML", "Accuracy": "0.6760 +/- 0.0298", "Precision": "0.6790 +/- 0.0304", "Recall": "0.9930 +/- 0.0054", "F1": "0.8060 +/- 0.0214", "F2": "0.9089 +/- 0.0144", "AUC": "0.5417 +/- 0.0444"},
    {"Horizon": "300 min", "Cohort": "MC-MED", "Model": "Ours", "Accuracy": "0.9229 +/- 0.0273", "Precision": "0.9802 +/- 0.0023", "Recall": "0.9401 +/- 0.0294", "F1": "0.9595 +/- 0.0151", "F2": "0.9478 +/- 0.0238", "AUC": "0.6847 +/- 0.1560"},
    {"Horizon": "300 min", "Cohort": "MC-MED", "Model": "CHA2DS2-VASc", "Accuracy": "0.7330 +/- 0.0474", "Precision": "0.9380 +/- 0.0161", "Recall": "0.7620 +/- 0.0446", "F1": "0.8410 +/- 0.0311", "F2": "0.7917 +/- 0.0409", "AUC": "0.5427 +/- 0.0807"},
    {"Horizon": "300 min", "Cohort": "MC-MED", "Model": "Nwosu-EHR RF", "Accuracy": "0.7790 +/- 0.0444", "Precision": "0.9690 +/- 0.0209", "Recall": "0.7870 +/- 0.0467", "F1": "0.8690 +/- 0.0316", "F2": "0.8177 +/- 0.0433", "AUC": "0.7031 +/- 0.0685"},
    {"Horizon": "300 min", "Cohort": "MC-MED", "Model": "Teoh-EHR XGB", "Accuracy": "0.9110 +/- 0.0217", "Precision": "0.9070 +/- 0.0242", "Recall": "0.9630 +/- 0.0337", "F1": "0.9340 +/- 0.0288", "F2": "0.9513 +/- 0.0317", "AUC": "0.4450 +/- 0.0771"},
    {"Horizon": "300 min", "Cohort": "MC-MED", "Model": "Yang-EHR ML", "Accuracy": "0.9300 +/- 0.0268", "Precision": "0.9210 +/- 0.0247", "Recall": "0.9740 +/- 0.0276", "F1": "0.9470 +/- 0.0260", "F2": "0.9629 +/- 0.0270", "AUC": "0.4542 +/- 0.0475"},
    {"Horizon": "360 min", "Cohort": "MIMIC-III", "Model": "Ours", "Accuracy": "0.8880 +/- 0.0028", "Precision": "0.8894 +/- 0.0013", "Recall": "0.9981 +/- 0.0025", "F1": "0.9406 +/- 0.0015", "F2": "0.9743 +/- 0.0020", "AUC": "0.6492 +/- 0.1147"},
    {"Horizon": "360 min", "Cohort": "MIMIC-III", "Model": "CHA2DS2-VASc", "Accuracy": "0.6400 +/- 0.0291", "Precision": "0.7950 +/- 0.0163", "Recall": "0.7380 +/- 0.0334", "F1": "0.7650 +/- 0.0222", "F2": "0.7487 +/- 0.0304", "AUC": "0.5149 +/- 0.0807"},
    {"Horizon": "360 min", "Cohort": "MIMIC-III", "Model": "Nwosu-EHR RF", "Accuracy": "0.7940 +/- 0.0265", "Precision": "0.7980 +/- 0.0265", "Recall": "0.9920 +/- 0.0059", "F1": "0.8840 +/- 0.0186", "F2": "0.9460 +/- 0.0117", "AUC": "0.4337 +/- 0.0630"},
    {"Horizon": "360 min", "Cohort": "MIMIC-III", "Model": "Teoh-EHR XGB", "Accuracy": "0.7970 +/- 0.0278", "Precision": "0.7930 +/- 0.0278", "Recall": "0.9890 +/- 0.0064", "F1": "0.8800 +/- 0.0196", "F2": "0.9424 +/- 0.0125", "AUC": "0.4792 +/- 0.0640"},
    {"Horizon": "360 min", "Cohort": "MIMIC-III", "Model": "Yang-EHR ML", "Accuracy": "0.7990 +/- 0.0278", "Precision": "0.8010 +/- 0.0278", "Recall": "0.9950 +/- 0.0048", "F1": "0.8880 +/- 0.0189", "F2": "0.9490 +/- 0.0113", "AUC": "0.4184 +/- 0.0532"},
    {"Horizon": "360 min", "Cohort": "MC-MED", "Model": "Ours", "Accuracy": "0.9797 +/- 0.0192", "Precision": "0.9975 +/- 0.0008", "Recall": "0.9804 +/- 0.0054", "F1": "0.9888 +/- 0.0025", "F2": "0.9837 +/- 0.0042", "AUC": "0.7079 +/- 0.0748"},
    {"Horizon": "360 min", "Cohort": "MC-MED", "Model": "CHA2DS2-VASc", "Accuracy": "0.7470 +/- 0.0441", "Precision": "0.9850 +/- 0.0008", "Recall": "0.7560 +/- 0.0444", "F1": "0.8550 +/- 0.0291", "F2": "0.7929 +/- 0.0393", "AUC": "0.7198 +/- 0.0573"},
    {"Horizon": "360 min", "Cohort": "MC-MED", "Model": "Nwosu-EHR RF", "Accuracy": "0.9560 +/- 0.0204", "Precision": "0.9880 +/- 0.0041", "Recall": "0.9770 +/- 0.0145", "F1": "0.9820 +/- 0.0107", "F2": "0.9792 +/- 0.0125", "AUC": "0.8221 +/- 0.3502"},
    {"Horizon": "360 min", "Cohort": "MC-MED", "Model": "Teoh-EHR XGB", "Accuracy": "0.0800 +/- 0.0293", "Precision": "1.0000 +/- 0.0000", "Recall": "0.0700 +/- 0.0268", "F1": "0.1300 +/- 0.0464", "F2": "0.0860 +/- 0.0323", "AUC": "0.1535 +/- 0.1955"},
    {"Horizon": "360 min", "Cohort": "MC-MED", "Model": "Yang-EHR ML", "Accuracy": "0.6900 +/- 0.0500", "Precision": "0.9840 +/- 0.0133", "Recall": "0.6980 +/- 0.0492", "F1": "0.8160 +/- 0.0352", "F2": "0.7411 +/- 0.0459", "AUC": "0.1023 +/- 0.2159"},
]


def format_markdown(rows: Iterable[dict[str, object]]) -> str:
    columns = ["Horizon", "Cohort", "Model", *METRICS]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---:|---|---|" + "|".join(["---:"] * len(METRICS)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def format_csv(rows: Iterable[dict[str, object]]) -> str:
    columns = ["Horizon", "Cohort", "Model", *METRICS]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the manuscript Table III benchmark reference values. "
            "This does not recompute restricted-data clinical/EHR baselines."
        )
    )
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional output file. Defaults to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = format_markdown(TABLE3_ROWS) if args.format == "markdown" else format_csv(TABLE3_ROWS)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
