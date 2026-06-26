"""
Ablation plot: DDM-BAE (DDM-3D) vs DDM-PCA across training set sizes.
Produces thesis/figures/ablation_baselines.pdf
"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "thesis/figures"

METRICS = {
    "shape_mse": "Shape MSE",
    "aoa_mse":   "AoA MSE",
    "vendi":     "Vendi Score",
}

FONT_LABEL  = 13
FONT_TICK   = 11
FONT_LEGEND = 12
FONT_XLABEL = 13

C_BAE = "steelblue"
C_PCA = "darkorange"


def load_jsons(directory, pattern):
    by_n = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        m = re.search(r"n(\d+)_s(\d+)", path)
        if not m:
            continue
        n = int(m.group(1))
        with open(path) as f:
            d = json.load(f)
        by_n[n].append(d)
    return by_n


def get_series(by_n, key):
    sizes, means, stds = [], [], []
    for n in sorted(by_n):
        vals = [d[key] for d in by_n[n] if key in d]
        if vals:
            sizes.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(sizes), np.array(means), np.array(stds)


def main():
    # DDM-BAE (DDM-3D) ablation JSONs
    by_n_bae = load_jsons("results/ddm_3d_ablation", "*/eval_*.json")
    # DDM-PCA ablation JSONs
    by_n_pca = load_jsons("results/ddm_pca_3d_ablation", "*/eval_*.json")

    n_metrics = len(METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols  # ceil

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.8, n_rows * 3.8))
    axes = axes.flatten()

    for ax, (metric_key, ylabel) in zip(axes, METRICS.items()):
        sizes_b, means_b, stds_b = get_series(by_n_bae, metric_key)
        sizes_p, means_p, stds_p = get_series(by_n_pca, metric_key)

        if len(sizes_b):
            ax.plot(sizes_b, means_b, marker="o", linewidth=1.8,
                    color=C_BAE, label="DDM-BAE")
            ax.fill_between(sizes_b, means_b - stds_b, means_b + stds_b,
                            alpha=0.25, color=C_BAE)
        if len(sizes_p):
            ax.plot(sizes_p, means_p, marker="o", linewidth=1.8,
                    color=C_PCA, label="DDM-PCA")
            ax.fill_between(sizes_p, means_p - stds_p, means_p + stds_p,
                            alpha=0.25, color=C_PCA)

        all_sizes = sorted(set(sizes_b.tolist()) | set(sizes_p.tolist()))
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_xticks(all_sizes)
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_TICK)
        ax.tick_params(axis="y", labelsize=FONT_TICK)

    # Hide any unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.supxlabel("Training dataset size", fontsize=FONT_XLABEL, y=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=FONT_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "ablation_baselines.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
