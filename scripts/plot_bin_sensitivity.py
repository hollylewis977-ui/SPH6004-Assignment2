from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.options import RESULTS_DIR
from utils.util import ensure_dir


METRICS = {
    "macro_c_index": ("Macro C-index", "Higher is better"),
    "macro_ibs": ("Macro IBS", "Lower is better"),
    "macro_nbll": ("Macro NBLL", "Lower is better"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bin-count sensitivity results from experiment directories.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment spec in the form <num_bins>:<result_dir_name>.",
    )
    parser.add_argument("--result-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "report" / "figures")
    parser.add_argument("--output-prefix", type=str, default="fig_bin_sensitivity_subsample")
    parser.add_argument("--title", type=str, default="Bin Sensitivity Under Subject-Level Train Subsampling")
    return parser.parse_args()


def _parse_experiment(spec: str) -> tuple[int, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid experiment spec: {spec}. Expected <num_bins>:<result_dir_name>.")
    num_bins_str, result_name = spec.split(":", 1)
    return int(num_bins_str), result_name.strip()


def _load_experiment_summary(result_root: Path, num_bins: int, result_name: str) -> dict[str, float | int | str]:
    summary_path = result_root / result_name / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Summary file is empty: {summary_path}")

    row: dict[str, float | int | str] = {"num_bins": int(num_bins), "experiment": result_name}
    for metric_name in METRICS:
        if metric_name not in summary_df.columns:
            raise ValueError(f"Metric {metric_name} missing from {summary_path}")
        values = summary_df[metric_name].to_numpy(dtype=float)
        row[f"{metric_name}_mean"] = float(np.mean(values))
        row[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return row


def main() -> None:
    args = parse_args()
    rows = [
        _load_experiment_summary(args.result_root, num_bins, result_name)
        for num_bins, result_name in (_parse_experiment(spec) for spec in args.experiment)
    ]
    summary_df = pd.DataFrame(rows).sort_values("num_bins").reset_index(drop=True)

    output_dir = ensure_dir(args.output_dir)
    summary_df.to_csv(output_dir / f"{args.output_prefix}.csv", index=False)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 3.8))
    if len(METRICS) == 1:
        axes = [axes]

    x = summary_df["num_bins"].to_numpy(dtype=int)
    for ax, (metric_name, (label, direction)) in zip(axes, METRICS.items()):
        means = summary_df[f"{metric_name}_mean"].to_numpy(dtype=float)
        stds = summary_df[f"{metric_name}_std"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color="#2F6DB3",
        )
        for bin_count, mean_value in zip(x, means):
            ax.text(bin_count, mean_value, f" {mean_value:.4f}", va="bottom", ha="left", fontsize=8)
        ax.set_xticks(x)
        ax.set_xlabel("Number of bins")
        ax.set_title(label)
        ax.set_ylabel(direction)
        ax.grid(alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(output_dir / f"{args.output_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{args.output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_string(index=False))
    print(f"Saved summary table and plots to {output_dir}")


if __name__ == "__main__":
    main()
