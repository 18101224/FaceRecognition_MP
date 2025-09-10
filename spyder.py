import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def parse_depth(feature_module_value: str) -> int:
    value = str(feature_module_value).strip().lower()
    if value in {"false", "none", "", "nan"}:
        return 0
    match = re.match(r"residual_(\d+)", value)
    if match:
        return int(match.group(1))
    return 0


def map_mode_from_freeze_backbone(freeze_value: str) -> str:
    # Adjusted: freeze_backbone == False => Fine-tuning (FT); True => Probing
    value = str(freeze_value).strip().lower()
    return "FT" if value == "false" else "Probing"


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "raf-db.csv"
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raf_ft_vs_probe.png"

    df = pd.read_csv(csv_path)

    # Ensure numeric type for best_acc
    df["best_acc"] = pd.to_numeric(df.get("best_acc"), errors="coerce")

    # Compute depth from feature_module
    df["depth"] = df.get("feature_module").apply(parse_depth)

    # Determine mode from freeze_backbone
    df["mode"] = df.get("freeze_backbone").apply(map_mode_from_freeze_backbone)

    # Aggregate: for each (mode, depth), pick max best_acc (handles multiple LRs)
    grouped = (
        df.dropna(subset=["best_acc"]).groupby(["mode", "depth"], as_index=False)["best_acc"].max()
    )

    # Pivot to have columns per mode for easy plotting
    pivot = grouped.pivot(index="depth", columns="mode", values="best_acc").sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    plotted_any = False
    for mode_name, style in [("FT", {"color": "tab:blue", "marker": "o"}), ("Probing", {"color": "tab:orange", "marker": "s"})]:
        if mode_name in pivot.columns:
            ax.plot(
                pivot.index.values,
                pivot[mode_name].values,
                label=mode_name,
                linewidth=2,
                markersize=6,
                **style,
            )
            plotted_any = True

    ax.set_xlabel("Feature module depth")
    ax.set_ylabel("Best accuracy")
    ax.set_title("RAF-DB: FT vs Probing by depth (max over LRs)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if plotted_any:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
    plt.show()