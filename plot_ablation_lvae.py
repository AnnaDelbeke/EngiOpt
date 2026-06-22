"""
Plot LVAE ablation: joint vs geom-only, metrics vs training size.

Usage:
    python plot_ablation_lvae.py
"""

import glob, json, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import numpy as np

JOINT_DIR    = "results/lvae_3d_ablation"
GEOM_DIR     = "results/lvae_3d_ablation_geom_only"
OUT_PATH     = "thesis/figures/ablation_lvae_metrics.pdf"

METRICS = {
    "shape_mse":    "Shape MSE",
    "pressure_mse": "Pressure MSE",
    "aoa_mse":      "AoA MSE",
    "mmd":          "Coord MMD",
    "mmd_w":        "Latent MMD",
}

C_JOINT = "steelblue"
C_GEOM  = "darkorange"


def load_by_n(ablation_dir):
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
            by_n[n].append(json.load(f))
    return by_n


def get_mean_std(by_n, metric):
    sizes, means, stds = [], [], []
    for n in sorted(by_n):
        vals = [r[metric] for r in by_n[n] if metric in r]
        if vals:
            sizes.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(sizes), np.array(means), np.array(stds)


def main():
    joint = load_by_n(JOINT_DIR)
    geom  = load_by_n(GEOM_DIR)

    metrics_to_plot = [k for k in METRICS
                       if any(k in r for runs in joint.values() for r in runs)]

    n_cols = 3
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.2, n_rows * 3.2))
    axes = np.array(axes).flatten()

    for ax, key in zip(axes, metrics_to_plot):
        for data, color, label, skip_keys in [
            (joint, C_JOINT, "Joint (geom + pressure)", set()),
            (geom,  C_GEOM,  "Geom-only",              {"pressure_mse"}),
        ]:
            if key in skip_keys:
                continue
            sizes, means, stds = get_mean_std(data, key)
            if len(sizes) == 0:
                continue
            ax.plot(sizes, means, marker="o", linewidth=1.8,
                    color=color, label=label)
            ax.fill_between(sizes, means - stds, means + stds,
                            alpha=0.20, color=color)
        ax.set_xlabel("Training dataset size", fontsize=9)
        ax.set_title(METRICS[key], fontsize=10)
        ax.set_xticks(sorted(joint.keys()))
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    # Legend in the empty axes slot to the right of Latent MMD
    empty_axes = [ax for ax in axes[len(metrics_to_plot):]]
    for ax in empty_axes:
        ax.set_visible(False)
    if empty_axes:
        legend_ax = empty_axes[0]
        legend_ax.set_visible(True)
        legend_ax.set_axis_off()
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, fontsize=10, loc="center",
                         frameon=False)

    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_PATH.replace(".pdf", f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()


if __name__ == "__main__":
    main()
