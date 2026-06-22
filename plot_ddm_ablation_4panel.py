"""
Reads 2D DDM ablation eval JSONs from results/evaluation/eval_n*_s*.json
and saves 4 individual PDFs (mmd, aoa_mse, vendi, shape_mse) with no title.
"""

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

EVAL_DIR = "results/evaluation"
OUT_DIR  = "results"

PANELS = [
    ("mmd",       "MMD (Wing Shape)"),
    ("aoa_mse",   "MSE (Angle of Attack)"),
    ("vendi",     "Vendi Score"),
    ("shape_mse", "Shape MSE"),
]


def load_data(eval_dir):
    by_n = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(eval_dir, "eval_n*_s*.json"))):
        m = re.search(r"eval_n(\d+)_s(\d+)\.json", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        with open(path) as f:
            d = json.load(f)
        by_n[n].append(d["results"])
    return by_n


def plot_panel(by_n, metric_key, ylabel, out_path):
    sizes = sorted(by_n.keys())
    means, stds, valid_sizes = [], [], []
    for n in sizes:
        vals = [run[metric_key]["mean"] for run in by_n[n] if metric_key in run]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_sizes.append(n)
    if not valid_sizes:
        print(f"No data for {metric_key}, skipping.")
        return

    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(valid_sizes, means, marker="o", linewidth=1.8, color="steelblue")
    ax.fill_between(valid_sizes, means - stds, means + stds, alpha=0.25, color="steelblue")
    ax.set_xlabel("Training dataset size", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(valid_sizes)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_n = load_data(EVAL_DIR)
    if not by_n:
        print(f"No eval JSONs found in {EVAL_DIR}")
        return
    print(f"Found sizes: {sorted(by_n.keys())}")
    for metric_key, ylabel in PANELS:
        out_path = os.path.join(OUT_DIR, f"ddm_ablation_{metric_key}.pdf")
        plot_panel(by_n, metric_key, ylabel, out_path)


if __name__ == "__main__":
    main()
