from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("imgs")
OUTPUT_FILE = OUTPUT_DIR / "model_comparison.png"

LEGACY_ACC = 91.33
Q_ACC = 89.15
LEGACY_LATENCY = 29.37
Q_LATENCY = 17.70
LEGACY_THROUGHPUT = 4357.6
Q_THROUGHPUT = 7231.9


def plot_metric(
    ax: plt.Axes,
    title: str,
    y_axis: str,
    legacy_value: float,
    quantized_value: float,
) -> None:
    labels = ["FP32 model", "Compressed model"]
    values = [legacy_value, quantized_value]
    colors = ["#4C78A8", "#F58518"]
    bars = ax.bar(labels, values, color=colors, width=0.55)

    ax.set_title(f"{title}: FP32 vs Compressed")
    ax.set_xlabel("Model")
    ax.set_ylabel(y_axis)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    max_value = max(values)
    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main() -> None:
    metrics = [
        ("accuracy", "%", LEGACY_ACC, Q_ACC),
        ("latency", "ms/batch (batch size 128)", LEGACY_LATENCY, Q_LATENCY),
        ("throughput", "img/s", LEGACY_THROUGHPUT, Q_THROUGHPUT),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, y_axis, legacy_value, quantized_value) in zip(axes, metrics):
        plot_metric(ax, title, y_axis, legacy_value, quantized_value)

    fig.suptitle("FP32 Model vs Compressed Model", fontsize=16, y=1.02)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
