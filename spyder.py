import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "result.csv")

    # X axis: epoch if present, else use row index
    if "epoch" in df.columns:
        x = df["epoch"].values
    else:
        x = df.index.values

    # ----- Figure 1: gradients -----
    fig1, ax1 = plt.subplots(figsize=(8, 4.8), dpi=150)
    if "backbone_grads" in df.columns:
        ax1.plot(x, df["backbone_grads"].values, label="backbone_grads", color="tab:blue", linewidth=2)
    if "weight_grads" in df.columns:
        ax1.plot(x, df["weight_grads"].values, label="weight_grads", color="tab:orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient magnitude")
    ax1.set_title("Gradients over epochs")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "grads.png")

    # ----- Figure 2: train_loss and its difference -----
    if "train_loss" in df.columns:
        train_loss = df["train_loss"].astype(float)
        diff = train_loss.diff().fillna(0.0)

        fig2, ax2 = plt.subplots(figsize=(8, 4.8), dpi=150)
        ax2.plot(x, train_loss.values, label="train_loss", color="tab:green", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Train loss", color="tab:green")
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.grid(True, linestyle="--", alpha=0.4)

        ax2b = ax2.twinx()
        ax2b.plot(x, diff.values, label="diff(train_loss)", color="tab:red", linewidth=1.8)
        ax2b.set_ylabel("Loss difference", color="tab:red")
        ax2b.tick_params(axis='y', labelcolor='tab:red')

        # Build a combined legend
        lines = ax2.get_lines() + ax2b.get_lines()
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="best")

        fig2.tight_layout()
        fig2.savefig(out_dir / "train_loss_and_diff.png")


if __name__ == "__main__":
    main()
    