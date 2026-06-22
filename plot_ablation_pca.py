"""
Plot DDM-PCA ablation metrics curves.

Reads eval JSON files from results/ddm_pca_3d_ablation/ and produces
a metrics-vs-training-size plot comparable to the DDM-W ablation figure.

Usage:
    python plot_ablation_pca.py
    python plot_ablation_pca.py --ablation_dir results/ddm_pca_3d_ablation \
                                --out thesis/figures/ablation_pca_metrics.pdf
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "shape_mse":        "Shape MSE",
    "mmd":              "Latent MMD",
    "explained_variance": "Explained Variance",
}


def find_eval_jsons(ablation_dir):
    by_n = defaultdict(list)
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s*"))):
        m = re.search(r"n(\d+)_s(\d+)", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        jsons = sorted(glob.glob(os.path.join(n_dir, "eval_*.json")))
        if not jsons:
            continue
        with open(jsons[-1]) as f:
            data = json.load(f)
        by_n[n].append(data)
    return by_n


def plot_metrics_curves(by_n, save_path):
    sizes = sorted(by_n.keys())
    metrics_to_plot = [k for k in METRICS if any(k in run for runs in by_n.values() for run in runs)]

    n_cols = 2
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.2))
    axes = np.array(axes).flatten()

    for ax, metric_key in zip(axes, metrics_to_plot):
        means, stds, valid_sizes = [], [], []
        for n in sizes:
            vals = [run[metric_key] for run in by_n[n] if metric_key in run]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_sizes.append(n)
        if not valid_sizes:
            ax.set_visible(False)
            continue
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(valid_sizes, means, marker="o", linewidth=1.8, color="steelblue")
        ax.fill_between(valid_sizes, means - stds, means + stds, alpha=0.25, color="steelblue")
        ax.set_xlabel("Training dataset size", fontsize=9)
        ax.set_ylabel(METRICS[metric_key], fontsize=9)
        ax.set_title(METRICS[metric_key], fontsize=10)
        ax.set_xticks(valid_sizes)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for ax in axes[len(metrics_to_plot):]:
        ax.set_visible(False)

    fig.suptitle("DDM-PCA Data Ablation", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_path.replace(".pdf", f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", default="results/pca_ablation")
    p.add_argument("--out", default="thesis/figures/ablation_pca_metrics.pdf")
    args = p.parse_args()

    by_n = find_eval_jsons(args.ablation_dir)
    if not by_n:
        print(f"No eval JSONs found in {args.ablation_dir}. Run eval first.")
        return
    print(f"Found data for n = {sorted(by_n.keys())}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_metrics_curves(by_n, args.out)


if __name__ == "__main__":
    main()
