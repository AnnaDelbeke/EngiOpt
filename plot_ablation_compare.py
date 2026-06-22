"""
Overlay the joint (geometry+pressure) and geometry-only DDM_W3D data
ablations on a single set of metric panels for direct comparison.

Usage:
    python plot_ablation_compare.py \
        [--joint_dir results/ddm_w_3d_ablation_v29] \
        [--geom_dir  results/ddm_w_3d_ablation_geom_only] \
        [--out results/ablation_compare_metrics.png]
"""

import argparse

from plot_ablation import METRICS, find_eval_jsons

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import numpy as np


COL_JOINT = "steelblue"
COL_GEOM  = "darkorange"


def series(by_n, metric_key):
    sizes = sorted(by_n.keys())
    means, stds, valid_sizes = [], [], []
    for n in sizes:
        vals = [run[metric_key] for run in by_n[n] if metric_key in run]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_sizes.append(n)
    return np.array(valid_sizes), np.array(means), np.array(stds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint_dir", default="results/ddm_w_3d_ablation_v29")
    p.add_argument("--geom_dir",  default="results/ddm_w_3d_ablation_geom_only")
    p.add_argument("--out",       default="results/ablation_compare_metrics.png")
    args = p.parse_args()

    by_n_joint = find_eval_jsons(args.joint_dir)
    by_n_geom  = find_eval_jsons(args.geom_dir)

    metrics_to_plot = [
        k for k in METRICS
        if any(k in run for runs in by_n_joint.values() for run in runs)
        or any(k in run for runs in by_n_geom.values()  for run in runs)
    ]

    n_cols = 3
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = np.array(axes).flatten()

    for ax, metric_key in zip(axes, metrics_to_plot):
        plotted = False

        sizes_j, means_j, stds_j = series(by_n_joint, metric_key)
        if len(sizes_j):
            ax.plot(sizes_j, means_j, marker="o", linewidth=1.8,
                    color=COL_JOINT, label="Joint (geometry + pressure)")
            ax.fill_between(sizes_j, means_j - stds_j, means_j + stds_j,
                             alpha=0.25, color=COL_JOINT)
            plotted = True

        # pressure_mse is only meaningful for the joint model
        if metric_key != "pressure_mse":
            sizes_g, means_g, stds_g = series(by_n_geom, metric_key)
            if len(sizes_g):
                ax.plot(sizes_g, means_g, marker="o", linewidth=1.8,
                        color=COL_GEOM, label="Geometry-only")
                ax.fill_between(sizes_g, means_g - stds_g, means_g + stds_g,
                                 alpha=0.25, color=COL_GEOM)
                plotted = True

        if not plotted:
            ax.set_visible(False)
            continue

        all_sizes = sorted(set(sizes_j.tolist()) | (
            set(sizes_g.tolist()) if metric_key != "pressure_mse" else set()
        ))
        ax.set_ylabel(METRICS[metric_key], fontsize=9)
        ax.set_xticks(all_sizes)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    for ax in axes[len(metrics_to_plot):]:
        ax.set_visible(False)

    fig.supxlabel("Training dataset size", fontsize=10, y=0.02)

    # Single legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
