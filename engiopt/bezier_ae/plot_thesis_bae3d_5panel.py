"""
Thesis-quality 5-panel reconstruction plot for the 3D (joint) BAE.

Shows 5 validation wings, each at the mid-span slice, with a shared y-axis
so airfoil proportions are directly comparable across panels.

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae3d_5panel
    python -m engiopt.bezier_ae.plot_thesis_bae3d_5panel \
        --checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_039 \
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
from torch.utils.data import random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

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


@torch.no_grad()
def plot_5panel(val_dataset, model, device, run_dir, wing_indices):
    S = val_dataset[0].shape[0]
    mid = S // 2

    results = []
    for wi in wing_indices:
        x = val_dataset[wi].unsqueeze(0).to(device)   # [1, S, 2, 192]
        z = model.encode(x)
        y, cp = model.decode(z, return_cp=True)

        x_s  = x[0, mid].cpu()    # [2, 192]
        y_s  = y[0, mid].cpu()
        cp_s = cp[0, mid].cpu()   # [2, n_cp]

        results.append((
            x_s[0].numpy(), x_s[1].numpy(),
            y_s[0].numpy(), y_s[1].numpy(),
            cp_s[0].numpy(), cp_s[1].numpy(),
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

    x_range = x_hi - x_lo
    y_range = y_hi - y_lo
    aspect = y_range / x_range

    fig_w    = 8.0
    leg_h    = 0.30
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
        path = os.path.join(save_dir, f"bae3d_5panel.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    scratch_dir = "/cluster/scratch/adelbeke/thesis_figures"
    os.makedirs(scratch_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(scratch_dir, f"bae3d_5panel.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir",    default="results/bezier_ae_3d/run_039")
    p.add_argument("--wing_indices", type=int, nargs="+", default=None)
    p.add_argument("--case_nums", type=int, nargs="+", default=None,
                   help="Include these case numbers in the 5-panel plot (looked up in the val set).")
    p.add_argument("--slices_pkl",  default=_SLICES_PKL)
    p.add_argument("--scalars_pkl", default=_SCALARS_PKL)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset ───────────────────────────────────────────────────────────────
    new_dataset = NewWingsDataset(args.slices_pkl, args.scalars_pkl, seed=0)
    all_items   = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                   if item["final"] == 1]
    all_case_nums = [item["case_num"] for item in all_items]
    full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=0)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    val_case_nums = [all_case_nums[i] for i in val_dataset.indices]
    print(f"Validation wings: {len(val_dataset)}")

    # Resolve indices from --case_nums and --wing_indices
    indices = list(args.wing_indices) if args.wing_indices else []
    if args.case_nums:
        for cn in args.case_nums:
            if cn in val_case_nums:
                indices.append(val_case_nums.index(cn))
            else:
                print(f"WARNING: case {cn} not found in val set — skipping")
    if not indices:
        indices = [0, 5, 10]
    # Deduplicate while preserving order, keep first 3
    seen = set()
    indices = [i for i in indices if not (i in seen or seen.add(i))][:3]
    print(f"Plotting val indices {indices} → cases {[val_case_nums[i] for i in indices]}")

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BezierAutoencoder3D(
        n_spans=ckpt["n_spans"],
        n_control_points=ckpt["n_control_points"],
        n_data_points=192,
        slice_hidden_dims=ckpt["slice_hidden_dims"],
        span_hidden_dims=ckpt["span_hidden_dims"],
        latent_dim=ckpt["latent_dim"],
        cpx_bound=ckpt["cpx_bound"],
        cpy_bound=ckpt["cpy_bound"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    plot_5panel(val_dataset, model, device, args.run_dir, indices)


if __name__ == "__main__":
    main()
