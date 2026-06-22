"""
Diagnostic figures for the axis-aligned latent traversal.

Figure A: w45 vs w46 scatter (training data) with the w45 traversal path overlaid.
          w45 = highest-variance active dim; w46 = lowest-variance active dim.
          These are the same two dimensions shown in the traversal figure.
Figure B: Distance to nearest training neighbour along the w45 traversal,
          computed in the full 20-dimensional active space.

Mirrors the checkpoint/data loading of plot_latent_dims.py exactly.

Usage:
    python plot_traversal_diagnostics.py
"""

import sys
import os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

import numpy as np
import torch

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d
from engiopt.lvae.plot_latent_tsne import collect_w_latents

BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

OUT_DIR  = "thesis/figures"
DEVICE   = "cpu"
N_STEPS  = 7
N_SIGMA  = 3.0

C_TRAIN = "#4477AA"
CMAP    = matplotlib.colors.LinearSegmentedColormap.from_list(
    "dark_div", ["#1a4f8a", "#6a3d9a", "#c0392b"]
)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading BAE ...")
    bae = load_bae_3d(BAE_CKP, DEVICE)

    print("Loading LVAE ...")
    lvae = load_lvae_3d(LVAE_CKP, DEVICE, bae)

    dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    train_items = [it for it in dataset["train"] if it["final"] == 1]
    test_items  = [it for it in dataset["test"]  if it["final"] == 1]
    print(f"Training wings: {len(train_items)},  test wings: {len(test_items)}")

    # ── encode training set ──────────────────────────────────────────────────
    print("Encoding training set ...")
    train_w = collect_w_latents(lvae, bae, train_items, DEVICE)  # [N_tr, 64]

    # Active dimensions sorted by descending variance (same logic as plot_latent_dims.py)
    train_std  = train_w.std(axis=0)
    active_idx = np.where(train_std >= 0.02)[0]
    active_idx = active_idx[np.argsort(train_std[active_idx])[::-1]]
    print(f"Active dims ({len(active_idx)}): {active_idx.tolist()}")

    # top = highest-variance active dim; bottom = lowest-variance active dim
    # These match the two dimensions shown in the traversal figure.
    idx_top1   = active_idx[0]    # w45  (highest variance)
    idx_bottom = active_idx[-1]   # w46  (lowest variance)

    # Column positions in the active-only sub-array (for NN search in 20-dim space)
    col_top1   = 0
    col_bottom = len(active_idx) - 1

    train_active = train_w[:, active_idx]  # [N_tr, 20]

    # ── reference wing: first transonic test wing (mirrors plot_latent_traversal.py) ──
    ref_item = next(it for it in test_items if 0.8 <= it["mach"] < 1.0)
    print(f"Reference wing: Mach={ref_item['mach']:.3f}, case={ref_item['case_num']}")

    z_bae, _gt, _aoa, params_scaled, _te, gt_pressure, _le, _ch = encode_item_3d(
        ref_item, bae, lvae, DEVICE, apply_x_norm=True,
    )
    with torch.no_grad():
        w_ref = lvae.encoder(
            z_bae.unsqueeze(0).to(DEVICE),
            gt_pressure.unsqueeze(0).to(DEVICE),
            params_scaled.to(DEVICE),
        )  # [1, 64]

    w_ref_np = w_ref.squeeze(0).cpu().numpy()  # [64]

    # ── build traversal vectors ──────────────────────────────────────────────
    # Sweep w45 (idx_top1) from -3sigma to +3sigma; all other dims fixed at reference.
    dim_std_top1 = float(train_std[idx_top1])
    vals = np.linspace(-N_SIGMA * dim_std_top1, N_SIGMA * dim_std_top1, N_STEPS)

    traversal_w = np.tile(w_ref_np, (N_STEPS, 1))   # [N_STEPS, 64]
    traversal_w[:, idx_top1] = vals

    traversal_active = traversal_w[:, active_idx]    # [N_STEPS, 20]
    sigma_vals = np.linspace(-N_SIGMA, N_SIGMA, N_STEPS)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Figure A: w45 vs w46 scatter + traversal path ────────────────────────
    # Determine the 1-based label for the bottom dim (lowest variance active dim)
    bottom_label = idx_bottom + 1   # e.g. dim index 45 -> "w46"

    print("Plotting Figure A ...")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.scatter(
        train_active[:, col_top1], train_active[:, col_bottom],
        s=6, alpha=0.35, color=C_TRAIN, label="Training set", rasterized=True,
    )

    step_colors = CMAP(np.linspace(0, 1, N_STEPS))
    for k in range(N_STEPS - 1):
        ax.plot(
            traversal_active[k:k+2, col_top1], traversal_active[k:k+2, col_bottom],
            color=step_colors[k], linewidth=2.0, zorder=3,
        )
    sc = ax.scatter(
        traversal_active[:, col_top1], traversal_active[:, col_bottom],
        c=np.linspace(0, 1, N_STEPS), cmap=CMAP,
        s=60, zorder=4, edgecolors="k", linewidths=0.5,
        label="Traversal steps",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels([f"$-{N_SIGMA:.0f}\\sigma$", "0", f"$+{N_SIGMA:.0f}\\sigma$"])
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(f"$w_{{45}}$")
    ax.set_ylabel(f"$w_{{{bottom_label}}}$")
    ax.legend(fontsize=11, markerscale=1.5, frameon=False)
    fig.tight_layout()

    path_a = os.path.join(OUT_DIR, "traversal_manifold_overlay.pdf")
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    fig.savefig(path_a.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_a}")

    # ── Figure B: NN distance along traversal in full 20-dim active space ────
    print("Computing nearest-neighbour distances ...")
    # [N_STEPS, N_tr, 20] -> min over training set -> [N_STEPS]
    diffs = train_active[np.newaxis, :, :] - traversal_active[:, np.newaxis, :]
    nn_dists = np.sqrt((diffs ** 2).sum(axis=-1)).min(axis=1)

    print("Plotting Figure B ...")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for k in range(N_STEPS - 1):
        ax.plot(
            sigma_vals[k:k+2], nn_dists[k:k+2],
            color=step_colors[k], linewidth=2.0,
        )
    ax.scatter(
        sigma_vals, nn_dists,
        c=np.linspace(0, 1, N_STEPS), cmap=CMAP,
        s=60, zorder=3, edgecolors="k", linewidths=0.5,
    )

    ax.set_xlabel(f"$w_{{45}}$ (multiples of $\\sigma$)")
    ax.set_ylabel("Distance to nearest\ntraining point")
    ax.set_xticks(sigma_vals)
    ax.set_xticklabels([f"{v:.1f}$\\sigma$" for v in sigma_vals], fontsize=10)
    fig.tight_layout()

    path_b = os.path.join(OUT_DIR, "traversal_nn_distance.pdf")
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    fig.savefig(path_b.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_b}")

    print("\nNN distances per step:")
    for k in range(N_STEPS):
        print(f"  w45 = {sigma_vals[k]:+.1f}sigma  ->  NN dist = {nn_dists[k]:.4f}")


if __name__ == "__main__":
    main()
