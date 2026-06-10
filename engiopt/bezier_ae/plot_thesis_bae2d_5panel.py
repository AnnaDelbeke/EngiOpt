"""
Thesis-quality 5-panel reconstruction plot for the 2D (per-slice) BAE.

All panels share the same y-axis so airfoil proportions are comparable.

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae2d_5panel
    python -m engiopt.bezier_ae.plot_thesis_bae2d_5panel \
        --checkpoint results/bezier_ae/run_006/models/bezier_ae_best.pt \
        --run_dir    results/bezier_ae/run_006 \
        --wing_indices 0 5 10 20 30
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engibench.problems.wings3D.v0 import Wings3D

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
})

COL_GT    = "#2166ac"
COL_RECON = "#d6604d"
COL_CP    = "#4dac26"
N_SLICES  = 9


@torch.no_grad()
def plot_5panel(val_dataset, model, device, run_dir, wing_indices):
    final_items = [item for item in val_dataset if item.get("final", 0) == 1]
    print(f"Final wings in val split: {len(final_items)}")

    results = []
    for wi in wing_indices:
        item   = final_items[wi]
        coords = np.array(item["coords"], dtype=np.float32)   # [9, 192, 2]
        x = torch.tensor(coords).permute(0, 2, 1)             # [9, 2, 192]
        for s in range(x.shape[0]):
            x[s, 1, :] -= x[s, 1, 0]
            x[s, 0, :] += (1.0 - x[s, 0, 0])

        # use mid-span slice for each example
        s_idx = N_SLICES // 2
        xi = x[s_idx].unsqueeze(0).to(device)
        y, _, _, cp, _ = model(xi)
        results.append((
            xi[0, 0].cpu().numpy(), xi[0, 1].cpu().numpy(),
            y[0, 0].cpu().numpy(),  y[0, 1].cpu().numpy(),
            cp[0, 0].cpu().numpy(), cp[0, 1].cpu().numpy(),
        ))

    # Shared y-limits: based on GT + reconstruction across ALL panels
    all_y = np.concatenate([r[1] for r in results] + [r[3] for r in results])
    y_margin = 0.05
    y_lo = all_y.min() - y_margin
    y_hi = all_y.max() + y_margin
    x_lo, x_hi = -0.02, 1.05

    n = len(results)
    fig, axes = plt.subplots(
        1, n,
        figsize=(2.6 * n, 2.6),
        sharey=True,          # ← shared y-axis, uniform scale across panels
        constrained_layout=True,
    )

    for col, ((gt_x, gt_y, recon_x, recon_y, cp_x, cp_y), ax) in \
            enumerate(zip(results, axes)):

        ax.plot(gt_x,    gt_y,    color=COL_GT,    lw=1.8, label="Ground truth", zorder=3)
        ax.plot(recon_x, recon_y, color=COL_RECON, lw=1.8, ls="--", label="Prediction", zorder=3)
        ax.plot(cp_x,    cp_y,    color=COL_CP,    lw=0.9, ls="--",
                marker="o", markersize=3.5, alpha=0.8, label="Control polygon", zorder=2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"({chr(ord('a') + col)}) Example {col + 1}", pad=3)
        ax.set_xlabel("x/c")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if col == 0:
            ax.set_ylabel("y/c")

    # Shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=3, framealpha=0.9, handlelength=1.8,
               bbox_to_anchor=(0.5, -0.08))

    save_dir = os.path.join(run_dir, "reconstructions_thesis")
    os.makedirs(save_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"bae2d_5panel.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    # also save to scratch for easy access
    scratch_dir = "/cluster/scratch/adelbeke/thesis_figures"
    os.makedirs(scratch_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(scratch_dir, f"bae2d_5panel.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/bezier_ae/run_006/models/bezier_ae_best.pt")
    p.add_argument("--run_dir",    default="results/bezier_ae/run_006")
    p.add_argument("--wing_indices", type=int, nargs="+", default=[0, 5, 10, 20, 30])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem   = Wings3D(seed=0)
    val_data  = problem.dataset["validation"]

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BezierAutoencoder(n_control_points=32, n_data_points=192, auto_batch=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded 2D BAE checkpoint.")

    plot_5panel(val_data, model, device, args.run_dir, args.wing_indices)


if __name__ == "__main__":
    main()
