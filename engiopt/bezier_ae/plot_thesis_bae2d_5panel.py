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

    # Shared y-limits: based on GT + reconstruction + control points across ALL panels
    # so every control point stays inside the plotted area
    all_y = np.concatenate([r[1] for r in results] + [r[3] for r in results] + [r[5] for r in results])
    all_x = np.concatenate([r[0] for r in results] + [r[2] for r in results] + [r[4] for r in results])
    y_margin = 0.05
    y_lo = all_y.min() - y_margin
    y_hi = all_y.max() + y_margin
    x_lo = min(-0.02, all_x.min() - y_margin)
    x_hi = max(1.05, all_x.max() + y_margin)

    # 1x4 horizontal layout with equal aspect panels
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo
    aspect = y_range / x_range

    fig_w    = 8.0
    leg_h    = 0.30   # fraction reserved at bottom for legend
    gap_x    = 0.04
    margin_l = 0.07
    margin_r = 0.01
    margin_t = 0.02

    ax_w = (1 - margin_l - margin_r - 2 * gap_x) / 3
    panel_w_in = ax_w * fig_w
    panel_h_in = panel_w_in * aspect
    fig_h = (panel_h_in / fig_w + leg_h + margin_t) * fig_w
    ax_h = panel_h_in / fig_h

    positions = [
        [margin_l + i * (ax_w + gap_x), leg_h, ax_w, ax_h]
        for i in range(3)
    ]

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes_flat = [fig.add_axes(pos) for pos in positions]

    for idx, ((gt_x, gt_y, recon_x, recon_y, cp_x, cp_y), ax) in \
            enumerate(zip(results, axes_flat)):

        ax.plot(gt_x,    gt_y,    color=COL_GT,    lw=1.2, label="Ground truth", zorder=3)
        ax.plot(recon_x, recon_y, color=COL_RECON, lw=1.2, ls="--", label="Reconstruction", zorder=3)
        ax.plot(cp_x,    cp_y,    color=COL_CP,    lw=0.6, ls="--",
                marker="o", markersize=2.5, alpha=0.8, label="Control polygon", zorder=2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if idx == 0:
            ax.set_ylabel("y/c")
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=3, framealpha=0.9, handlelength=1.2, fontsize=9,
               bbox_to_anchor=(0.5, 0.05), borderpad=0.6,
               columnspacing=1.0)

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
    p.add_argument("--wing_indices", type=int, nargs="+", default=[0, 5, 10])
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
