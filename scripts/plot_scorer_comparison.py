"""
Plot candidate scorer comparison chart (Figure 5 in Research Plan).
Horizontal bar plot comparing Discrimination Delta and Latency across evaluated candidate models.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_scorer_plot(
    csv_path: str = "results/scorer_benchmark_registry.csv",
    output_png: str = "plots/fig5_scorer_comparison.png",
):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    # Drop entries without delta_score
    df = df.dropna(subset=["delta_score"])
    if len(df) == 0:
        print("No evaluated scorers found.")
        return

    df = df.sort_values(by="delta_score", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Bar colors by status
    colors = ["#5cb85c" if s == "VERIFIED_ACTIVE" else "#f0ad4e" for s in df["status"]]

    # 1. Delta Discrimination Bar Plot
    model_labels = [m.split("/")[-1] for m in df["model_id"]]
    ax1.barh(model_labels, df["delta_score"], color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xlim(-0.05, 1.05)
    ax1.axvline(0.2, color="red", linestyle="--", alpha=0.6, label="Active Cutoff (0.2)")
    ax1.set_xlabel("Toxicity Discrimination Delta (Toxic - Safe Score)", fontsize=10)
    ax1.set_title("Safety Discrimination Capability", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Latency Comparison
    lat_colors = ["#0275d8" if s == "VERIFIED_ACTIVE" else "#999999" for s in df["status"]]
    ax2.barh(model_labels, df["latency_ms"], color=lat_colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Inference Latency per Sample (ms)", fontsize=10)
    ax2.set_title("Per-Step Inference Latency", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 5: Candidate Safety Scorer Registry Benchmark (Section 5.2)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"Generated Figure 5 -> {output_png}")


if __name__ == "__main__":
    generate_scorer_plot()
