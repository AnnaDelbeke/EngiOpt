"""
Thesis figures for the 3D joint BAE — error-along-the-wing and MSE distribution.

Figure 1: Per-span reconstruction MSE (x1e-6) vs. physical span position
           z (root = 0, tip ~= 2.25), averaged over the validation set.
           Note: the dataset's "eta"/"transforms" field is the physical
           span coordinate z, NOT a normalized 0-1 fraction.
Figure 2: Histogram + KDE of per-wing reconstruction MSE (x1e-6), one value
           per validation wing (averaged over its spans) — shows whether
           reconstruction quality is consistent across wings or has outliers.

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae3d_error_density
    python -m engiopt.bezier_ae.plot_thesis_bae3d_error_density \
        --checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_039
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
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

COL_MSE  = "#2166ac"
COL_KDE  = "#d6604d"


def _save(fig, run_dir, stem):
    save_dir = os.path.join(run_dir, "reconstructions_thesis")
    os.makedirs(save_dir, exist_ok=True)
    scratch_dir = "/cluster/scratch/adelbeke/thesis_figures"
    os.makedirs(scratch_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        for base in (save_dir, scratch_dir):
            path = os.path.join(base, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"Saved: {path}")


@torch.no_grad()
def plot_error_along_wing(per_span_mse, z_positions, run_dir):
    """Per-span MSE (x1e-6) vs. physical span position z (root=0, tip~=2.25)."""
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)

    ax.plot(z_positions, per_span_mse * 1e6, color=COL_MSE, lw=1.8, marker="o", markersize=4)
    ax.set_xlabel(r"Span position $z$ (root = 0, tip $\approx$ 2.25)")
    ax.set_ylabel(r"Reconstruction MSE ($\times 10^{-6}$)")
    ax.set_xlim(-0.05, z_positions.max() + 0.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, run_dir, "bae3d_error_along_span")
    plt.close()


@torch.no_grad()
def plot_mse_density(per_wing_mse, run_dir):
    """Histogram + KDE of per-wing reconstruction MSE (x1e-6)."""
    vals = per_wing_mse * 1e6

    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)

    n_bins = max(8, int(np.sqrt(len(vals))))
    counts, bin_edges, _ = ax.hist(
        vals, bins=n_bins, color="#f4a9a8", alpha=0.9,
        edgecolor="black", linewidth=0.8,
        label="3D Dataset",
    )
    bin_width = bin_edges[1] - bin_edges[0]

    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    # scale KDE (a density) to the same "number of wings" units as the histogram
    ax.plot(xs, kde(xs) * len(vals) * bin_width, color="red", lw=1.5)

    ax.set_xlabel(r"Reconstruction MSE ($\times 10^{-6}$)")
    ax.set_ylabel("Number of wings")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, run_dir, "bae3d_mse_density")
    plt.close()

    print(f"\nPer-wing MSE stats (x1e-6): "
          f"mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir",    default="results/bezier_ae_3d/run_039")
    p.add_argument("--slices_pkl", default=_SLICES_PKL)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset (same split as training: seed=0, 90/10) ────────────────────
    new_dataset = NewWingsDataset(args.slices_pkl, _SCALARS_PKL, seed=0)
    all_items   = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                   if item["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items)

    # physical span position z per item (the "transforms" field; despite being
    # called eta elsewhere, it is NOT normalized -- it ranges ~0.01 to ~2.5),
    # sorted root->tip, same order as coords
    all_z = [item["transforms"] for item in all_items]

    train_size = int(0.9 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    val_z = np.stack([all_z[i] for i in val_dataset.indices])  # [N_val, S]
    print(f"Validation wings: {len(val_dataset)}")

    # ── model ────────────────────────────────────────────────────────────
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
    print(f"Loaded model (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    n_spans = val_dataset[0].shape[0]
    per_span_mse = np.zeros(n_spans)
    per_wing_mse = np.zeros(len(val_dataset))

    with torch.no_grad():
        for i in range(len(val_dataset)):
            x = val_dataset[i].unsqueeze(0).to(device)
            y, _ = model(x)
            sq = ((x - y) ** 2).squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # [S]
            per_span_mse += sq
            per_wing_mse[i] = sq.mean()

    per_span_mse /= len(val_dataset)
    mean_z = val_z.mean(axis=0)  # average z per span index across wings

    plot_error_along_wing(per_span_mse, mean_z, args.run_dir)
    plot_mse_density(per_wing_mse, args.run_dir)


if __name__ == "__main__":
    main()
