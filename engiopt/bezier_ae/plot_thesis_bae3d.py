"""
Thesis-quality plots for the 3D BAE:
  1. Reconstruction grid (all 15 slices, GT + recon + control polygon)
  2. Aggregate spanwise MSE bar chart over the full validation set

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae3d
    python -m engiopt.bezier_ae.plot_thesis_bae3d \
        --checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_040 \
        --n_examples 3
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

COL_GT    = "#2166ac"   # blue
COL_RECON = "#d6604d"   # red-orange
COL_CP    = "#4dac26"   # green


# ── reconstruction grid ───────────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstruction_grid(x_wing: torch.Tensor,
                             y_wing: torch.Tensor,
                             cp_wing: torch.Tensor,
                             example_idx: int,
                             run_dir: str):
    """All S slices in a 3-column grid. Each subplot shows GT, reconstruction,
    and the Bézier control polygon.

    x_wing / y_wing : [S, 2, 192]
    cp_wing         : [S, 2, n_cp]
    """
    S     = x_wing.shape[0]
    ncols = 3
    nrows = (S + ncols - 1) // ncols

    # Give each subplot enough room so the airfoil is readable
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 2.2 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    slice_labels = (
        ["Root"] +
        [f"Slice {s+1}" for s in range(1, S - 1)] +
        ["Tip"]
    )

    # Compute shared x/y limits across all slices + control points
    all_y = np.concatenate([x_wing[:, 1].numpy().ravel(),
                             y_wing[:, 1].numpy().ravel(),
                             cp_wing[:, 1].numpy().ravel()])
    y_margin = 0.04
    y_lo = all_y.min() - y_margin
    y_hi = all_y.max() + y_margin
    x_lo, x_hi = -0.02, 1.02

    for s in range(S):
        ax = axes[s]

        gt_x    = x_wing[s, 0].numpy()
        gt_y    = x_wing[s, 1].numpy()
        recon_x = y_wing[s, 0].numpy()
        recon_y = y_wing[s, 1].numpy()
        cp_x    = cp_wing[s, 0].numpy()
        cp_y    = cp_wing[s, 1].numpy()

        ax.plot(gt_x,    gt_y,    color=COL_GT,    lw=1.8,
                label="Ground truth", zorder=3)
        ax.plot(recon_x, recon_y, color=COL_RECON, lw=1.8, ls="--",
                label="Reconstruction", zorder=3)
        ax.plot(cp_x,    cp_y,    color=COL_CP,    lw=0.9, ls="--",
                marker="o", markersize=4, alpha=0.85,
                label="Control polygon", zorder=2)

        ax.set_title(slice_labels[s], pad=3)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if s == 0:
            ax.legend(loc="upper right", framealpha=0.9,
                      handlelength=1.6, borderpad=0.4)

    for s in range(S, len(axes)):
        axes[s].set_visible(False)

    fig.suptitle(f"3D BAE — wing reconstruction (all {S} spanwise slices)",
                 fontsize=11, y=1.01)

    save_dir = os.path.join(run_dir, "reconstructions_thesis")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"wing_{example_idx:02d}_reconstruction_thesis.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── spanwise MSE bar chart ───────────────────────────────────────────────────

@torch.no_grad()
def plot_aggregate_spanwise_mse(val_dataset, model, device: torch.device,
                                run_dir: str):
    """Mean ± std reconstruction MSE per spanwise slice over all val wings."""
    S = val_dataset[0].shape[0]
    mse_matrix = []   # will be [N_val, S]

    for i in range(len(val_dataset)):
        x = val_dataset[i].unsqueeze(0).to(device)   # [1, S, 2, 192]
        y, _ = model(x)
        mse_per_slice = ((x - y) ** 2).mean(dim=(2, 3))  # [1, S]
        mse_matrix.append(mse_per_slice.squeeze(0).cpu().numpy())

    mse_matrix = np.stack(mse_matrix)          # [N_val, S]
    means = mse_matrix.mean(axis=0)            # [S]
    stds  = mse_matrix.std(axis=0)            # [S]

    # Colour the two worst slices
    worst2 = set(np.argsort(means)[-2:])
    colours = [COL_RECON if i in worst2 else COL_GT for i in range(S)]

    x_pos   = np.arange(1, S + 1)
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)

    ax.bar(x_pos, means, color=colours, alpha=0.85, zorder=3)
    ax.errorbar(x_pos, means, yerr=stds,
                fmt="none", color="#555555", capsize=3, lw=1.2, zorder=4)

    ax.set_xlabel("Spanwise position  (1 = root, 15 = tip)")
    ax.set_ylabel("Mean reconstruction MSE")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ["Root"] + [str(i) for i in range(2, S)] + ["Tip"],
        rotation=0,
    )
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True)
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Manual legend patch for the colour coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COL_GT,    alpha=0.85, label="Spanwise slice"),
        Patch(facecolor=COL_RECON, alpha=0.85, label="Highest MSE"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.9, handlelength=1.2)

    n_wings = len(val_dataset)
    ax.set_title(
        f"3D BAE — spanwise reconstruction MSE  "
        f"(mean ± std over {n_wings} validation wings)",
        pad=6,
    )

    save_dir = os.path.join(run_dir, "reconstructions_thesis")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "aggregate_spanwise_mse_thesis.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir",    type=str,
                   default="results/bezier_ae_3d/run_040")
    p.add_argument("--n_examples", type=int, default=3,
                   help="Number of wing reconstructions to plot")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset ──────────────────────────────────────────────────────────────
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
    n_spans           = ckpt.get("n_spans", val_dataset[0].shape[0])
    n_control_points  = ckpt.get("n_control_points", 32)
    slice_hidden_dims = ckpt.get("slice_hidden_dims", [64, 32])
    span_hidden_dims  = ckpt.get("span_hidden_dims",  [64, 32])
    latent_dim        = ckpt.get("latent_dim", 64)
    cpx_bound         = ckpt.get("cpx_bound", [0.0, 1.0])
    cpy_bound         = ckpt.get("cpy_bound", [-0.75, 0.75])

    model = BezierAutoencoder3D(
        n_spans=n_spans,
        n_control_points=n_control_points,
        n_data_points=192,
        slice_hidden_dims=slice_hidden_dims,
        span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim,
        cpx_bound=cpx_bound,
        cpy_bound=cpy_bound,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded run_040  (latent_dim={latent_dim}, n_spans={n_spans})")

    # ── reconstruction grids ─────────────────────────────────────────────────
    for i in range(min(args.n_examples, len(val_dataset))):
        x  = val_dataset[i].unsqueeze(0).to(device)   # [1, S, 2, 192]
        z  = model.encode(x)
        y, cp = model.decode(z, return_cp=True)

        x_wing  = x.squeeze(0).cpu()    # [S, 2, 192]
        y_wing  = y.squeeze(0).cpu()
        cp_wing = cp.squeeze(0).cpu()   # [S, 2, n_cp]

        mse = ((x_wing - y_wing) ** 2).mean().item()
        print(f"Wing {i+1} MSE: {mse:.2e}")

        plot_reconstruction_grid(x_wing, y_wing, cp_wing, i + 1, args.run_dir)

    # ── spanwise MSE ─────────────────────────────────────────────────────────
    plot_aggregate_spanwise_mse(val_dataset, model, device, args.run_dir)

    print(f"\nAll plots saved to {args.run_dir}/reconstructions_thesis/")


if __name__ == "__main__":
    import matplotlib
    main()
