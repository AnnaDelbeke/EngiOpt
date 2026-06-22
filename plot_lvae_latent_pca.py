"""
PCA projection of the LVAE latent space, coloured by Mach number.

Projects the 20 active LVAE latent dimensions onto their top two principal
components and plots a scatter coloured by continuous Mach number.
This gives a linear, distortion-free view of whether the latent space
organises by flow regime without relying on t-SNE.

Usage
-----
    python plot_lvae_latent_pca.py \
        --checkpoint results/lvae_3d/run_v29/lvae_3d_v29_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --save_dir results/plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from sklearn.decomposition import PCA

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.train_lvae_3d import LVAE3D
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
from engiopt.lvae.plot_latent_tsne import collect_w_latents

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--bae_checkpoint", type=str,
                   default="results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt")
    p.add_argument("--save_dir",       type=str, default="results/plots")
    p.add_argument("--split",          type=str, default="train",
                   choices=["train", "test", "val"],
                   help="Which dataset split to encode (default: train for better coverage)")
    p.add_argument("--seed",           type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bae_model = load_bae_3d(args.bae_checkpoint, device)
    model     = load_lvae_3d(args.checkpoint, device, bae_model)

    dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    items   = [it for it in dataset[args.split] if it["final"] == 1]
    print(f"Encoding {len(items)} wings from '{args.split}' split...")

    # Encode through LVAE encoder → w-vectors [N, lae_latent_dim]
    w_all = collect_w_latents(model, bae_model, items, device)

    # Apply the active-dimension mask (zeros out dead dimensions)
    if hasattr(model, "active_latent_mask") and model.active_latent_mask is not None:
        mask   = model.active_latent_mask.cpu().numpy().astype(bool)
        w_active = w_all[:, mask]
        n_active = mask.sum()
    else:
        w_active = w_all
        n_active = w_all.shape[1]
    print(f"Active latent dimensions: {n_active}")

    # PCA to 2D on the active dimensions
    pca   = PCA(n_components=2, random_state=0)
    w_2d  = pca.fit_transform(w_active)
    var   = pca.explained_variance_ratio_
    print(f"Variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={sum(var):.1%}")

    machs = np.array([it["mach"] for it in items])

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = mcolors.Normalize(vmin=machs.min(), vmax=machs.max())
    sc   = ax.scatter(w_2d[:, 0], w_2d[:, 1],
                      c=machs, cmap="coolwarm", norm=norm,
                      s=20, alpha=0.8, edgecolors="none")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mach number", fontsize=10)

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)", fontsize=10)
    ax.set_title(f"LVAE latent space — PCA projection\n"
                 f"({n_active} active dims, {args.split} set, $n={len(items)}$)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, "lvae_latent_pca_mach.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
