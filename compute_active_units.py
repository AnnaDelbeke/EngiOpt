"""
Compute active latent units for a set of LVAE checkpoints.

Active unit definition (Burda et al. 2016):
    AU_d = 1  iff  Var_{x ~ test}[ E_q[w_d | x] ] >= threshold

We also report the per-dimension std and KL proxy (std of w across dataset)
so the full profile can be compared across runs.

Usage:
    python compute_active_units.py
"""

import sys
import os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

import numpy as np
import torch

from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
from engiopt.lvae.plot_latent_tsne import collect_w_latents
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

BAE_CKP = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

CHECKPOINTS = {
    "v28":              "results/lvae_3d/lvae_3d_v28_best.pth",
    "v29":              "results/lvae_3d/lvae_3d_v29_best.pth",
    "v29_original":     "results/lvae_3d/lvae_3d_v29_original_best.pth",
    "v29_reproducing":  "results/lvae_3d/lvae_3d_v29_reproducing_best.pth",
    "v30":              "results/lvae_3d/lvae_3d_v30_best.pth",
}

AU_THRESHOLD = 0.01
DEVICE = "cpu"


def compute_au(w: np.ndarray, threshold: float = AU_THRESHOLD):
    """w: [N, D]. Returns per-dim std and boolean active mask."""
    std = w.std(axis=0)          # [D]  — Var_x[E_q[w_d|x]]^0.5
    active = std >= threshold
    return std, active


def main():
    bae = load_bae_3d(BAE_CKP, DEVICE)

    dataset    = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    test_items = [it for it in dataset["test"] if it["final"] == 1]
    print(f"Test wings: {len(test_items)}\n")

    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"Loading {name} …")
        lvae = load_lvae_3d(ckpt_path, DEVICE, bae)
        w    = collect_w_latents(lvae, bae, test_items, DEVICE)   # [N, 64]
        std, active = compute_au(w)
        n_active    = int(active.sum())
        saved_mask  = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False).get("active_latent_mask")
        n_saved = int(saved_mask.sum().item()) if saved_mask is not None else None
        results[name] = dict(w=w, std=std, active=active,
                             n_active=n_active, n_saved=n_saved)
        print(f"  AU (test, threshold={AU_THRESHOLD}): {n_active}/64  "
              f"| saved at training: {n_saved}/64")

    # ── summary table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Run':<22} {'AU (test)':>10} {'AU (saved)':>12} {'mean std':>10} {'max std':>10}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<22} {r['n_active']:>10}/64 "
              f"{str(r['n_saved'])+'/64':>12} "
              f"{r['std'].mean():>10.4f} "
              f"{r['std'].max():>10.4f}")

    # ── per-dimension std for the best run (v29) ─────────────────────────
    print("\nPer-dimension std for v29 (sorted descending):")
    std_v29   = results["v29"]["std"]
    active_v29 = results["v29"]["active"]
    order = np.argsort(std_v29)[::-1]
    for rank, d in enumerate(order[:25]):
        marker = "*" if active_v29[d] else " "
        print(f"  {marker} dim {d:2d}  std={std_v29[d]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
