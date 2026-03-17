from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NO_MP_LABEL = "w.o. MP"
MP_LABEL = "w/ MP"

NO_MP_COLOR = "#1f77b4"
MP_COLOR = "#d95f02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mixed precision vs no mixed precision comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/mp_vs_nompi_comparison.png",
        help="Path to the output image.",
    )
    return parser.parse_args()


def format_runtime(hours: float) -> str:
    total_minutes = int(round(hours * 60.0))
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour}h {minute:02d}m"


def annotate_bars(ax, bars, fmt, offset=0.12):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = ["LFW", "AgeDB-30", "CFP-FP", "CPLFW", "CALFW", "Mean"]
    no_mp_acc = np.array([99.1167, 92.7500, 95.5429, 89.1000, 92.4000, 93.7819], dtype=np.float64)
    mp_acc = np.array([99.0000, 90.5833, 94.5857, 88.2833, 91.6000, 92.8105], dtype=np.float64)

    no_mp_runtime_hours = 13.0
    mp_runtime_hours = 8.0 + 47.0 / 60.0

    no_mp_throughput = 560.0
    mp_throughput = 790.0

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.05], height_ratios=[1.0, 1.0])

    accuracy_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    ax_runtime = fig.add_subplot(grid[0, 3])
    ax_throughput = fig.add_subplot(grid[1, 3])

    for index, ax in enumerate(accuracy_axes):
        values = [no_mp_acc[index], mp_acc[index]]
        bars = ax.bar(
            [NO_MP_LABEL, MP_LABEL],
            values,
            color=[NO_MP_COLOR, MP_COLOR],
            width=0.55,
        )
        ax.set_title(datasets[index], fontsize=13, pad=8)
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
        annotate_bars(ax, bars, lambda value: f"{value:.2f}", offset=1.1)

    runtime_values = np.array([no_mp_runtime_hours, mp_runtime_hours], dtype=np.float64)
    runtime_labels = [NO_MP_LABEL, MP_LABEL]
    runtime_bars = ax_runtime.bar(runtime_labels, runtime_values, color=[NO_MP_COLOR, MP_COLOR], width=0.55)
    ax_runtime.set_title("Training Runtime", fontsize=14, pad=10)
    ax_runtime.set_ylabel("Hours")
    ax_runtime.grid(axis="y", alpha=0.25)
    annotate_bars(ax_runtime, runtime_bars, lambda value: format_runtime(value), offset=0.12)
    runtime_reduction = 100.0 * (no_mp_runtime_hours - mp_runtime_hours) / no_mp_runtime_hours
    ax_runtime.text(
        0.5,
        max(runtime_values) * 0.92,
        f"MP runtime reduced by {runtime_reduction:.1f}%",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f3f3f3", "edgecolor": "#cccccc"},
    )

    throughput_values = np.array([no_mp_throughput, mp_throughput], dtype=np.float64)
    throughput_bars = ax_throughput.bar(runtime_labels, throughput_values, color=[NO_MP_COLOR, MP_COLOR], width=0.55)
    ax_throughput.set_title("Training Throughput", fontsize=14, pad=10)
    ax_throughput.set_ylabel("Images / s")
    ax_throughput.grid(axis="y", alpha=0.25)
    annotate_bars(ax_throughput, throughput_bars, lambda value: f"{value:.0f}/s", offset=10.0)
    throughput_gain = mp_throughput / no_mp_throughput
    ax_throughput.text(
        0.5,
        max(throughput_values) * 0.90,
        f"MP throughput: {throughput_gain:.2f}x",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f3f3f3", "edgecolor": "#cccccc"},
    )

    fig.suptitle("KP-RPE: Mixed Precision vs No Mixed Precision", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
