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

    # Shared y-limits from GT + reconstruction only (control polygon can fly off)
    all_y = np.concatenate([r[1] for r in results] + [r[3] for r in results])
    y_margin = 0.05
    y_lo = all_y.min() - y_margin
    y_hi = all_y.max() + y_margin
    x_lo, x_hi = -0.02, 1.05

    n = len(results)
    fig, axes = plt.subplots(
        1, n,
        figsize=(2.6 * n, 2.6),
        sharey=True,
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

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=3, framealpha=0.9, handlelength=1.8,
               bbox_to_anchor=(0.5, -0.08))

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
    p.add_argument("--wing_indices", type=int, nargs="+", default=[0, 5, 10, 20, 30])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset ───────────────────────────────────────────────────────────────
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items   = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                   if item["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=0)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Validation wings: {len(val_dataset)}")

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
    print(f"Loaded run_039 (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    plot_5panel(val_dataset, model, device, args.run_dir, args.wing_indices)


if __name__ == "__main__":
    main()
