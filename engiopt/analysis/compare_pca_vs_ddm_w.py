"""
Side-by-side metric comparison: DDM_PCA vs DDM_W_3D (v29+).

Both models are run on the same test split and evaluated with the same metrics:
  shape_mse, aoa_mse, mmd, vendi, pressure_mse (DDM_W only), mmd_z / mmd_w.

DDM_PCA uses the 2D BezierAutoencoder (bezier_ae_best.pt).
DDM_W uses the 3D BezierAutoencoder3D (run_039).

Usage
-----
    python -m engiopt.analysis.compare_pca_vs_ddm_w \
        --ddm_pca_checkpoint  results/ddm_pca/ddm_pca_v1_best.pth \
        --ddm_w_checkpoint    results/ddm_w_3d/ddm_w_3d_v29_best.pth \
        --bae_3d_checkpoint   results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint     results/lvae_3d/lvae_3d_v29_best.pth \
        [--n_passes 10] [--seed 0] [--out_dir results/evaluation]
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_pca.ddm_pca import DDM_PCA, MLPDenoiser as PCADenoiser
from engiopt.ddm.ddm_pca.train_ddm_pca import (
    Config as PCAConfig, load_bae, build_sampler as build_sampler_pca,
    precompute_bae_latents,
)
from engiopt.ddm.ddm_pca.evaluate_ddm_pca import (
    precompute_test as precompute_test_pca,
    compute_metrics as compute_metrics_pca,
)
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser as WDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config as WConfig, load_lvae_3d, build_sampler as build_sampler_w
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import (
    precompute_test_3d,
    compute_metrics as compute_metrics_w,
)
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ddm_pca(args, device):
    cfg = PCAConfig()
    bae_2d = load_bae(cfg)  # 2D BezierAutoencoder from bezier_ae_best.pt

    pca_path = args.ddm_pca_checkpoint.replace("_best.pth", "_pca.pkl").replace(".pth", "_pca.pkl")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    ckpt  = torch.load(args.ddm_pca_checkpoint, map_location="cpu", weights_only=False)
    z_dim = ckpt.get("z_dim", cfg.n_components)
    pms   = ckpt.get("params_mean_std")
    ams   = ckpt.get("aoas_mean_std")

    denoiser = PCADenoiser(z_dim=z_dim, c_dim=cfg.c_dim).to(device)
    model = DDM_PCA(
        denoiser=denoiser, pca=pca, bae_model=bae_2d,
        sampler=build_sampler_pca(cfg), z_dim=z_dim, c_dim=cfg.c_dim,
        n_slices=cfg.n_slices,
        bae_latent_channels=cfg.bae_latent_channels,
        bae_latent_length=cfg.bae_latent_length,
        w_aoa=ckpt.get("w_aoa", 1.0),
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.ddm_pca_checkpoint))[0],
    )
    model.load(args.ddm_pca_checkpoint, train_mode=False)
    model.denoiser.to(device)
    print("DDM_PCA loaded (2D BAE).")
    return model, pca, pms, ams, bae_2d


def load_ddm_w(args, device):
    cfg = WConfig()
    cfg.bae_checkpoint  = args.bae_3d_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    bae_3d     = load_bae_3d(args.bae_3d_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_3d)
    ckpt       = torch.load(args.ddm_w_checkpoint, map_location="cpu", weights_only=False)
    sampler    = build_sampler_w(cfg)

    w_dim = cfg.lae_latent_dim
    saved = ckpt["denoiser"]
    if isinstance(saved, WDenoiser):
        denoiser = saved
    else:
        denoiser = WDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)

    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")

    model = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_3d,
        sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.ddm_w_checkpoint))[0],
    )
    model.w_mean     = ckpt.get("w_mean")
    model.w_std      = ckpt.get("w_std")
    model.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    model.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    model.denoiser.to(device)
    print("DDM_W3D loaded (3D BAE run_039).")
    return model, bae_3d, lvae_model, pms, ams


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ddm_pca_checkpoint", type=str, required=True)
    p.add_argument("--ddm_w_checkpoint",   type=str, required=True)
    p.add_argument("--bae_3d_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint",    type=str, required=True)
    p.add_argument("--n_passes", type=int, default=N_FORWARD_PASSES)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--T",        type=int, default=None)
    p.add_argument("--out_dir",  type=str, default="results/evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Load both models (each with their own BAE)
    ddm_pca, pca, pms_pca, ams_pca, bae_2d = load_ddm_pca(args, device)
    ddm_w, bae_3d, lvae_model, pms_w, ams_w = load_ddm_w(args, device)

    # Dataset
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    # ── DDM_PCA precompute (2D BAE) ───────────────────────────────────────────
    print("\nPrecomputing GT for DDM_PCA...")
    cfg_pca = PCAConfig()
    gt_coords_pca, gt_aoas_pca, gt_z_pca, z_inits_pca, params_pca = precompute_test_pca(
        test_dataset, initial_by_case, bae_2d, pca,
        ddm_pca.z_mean, ddm_pca.z_std,
        ddm_pca.scaler_params, ddm_pca.scaler_aoas,
        cfg_pca, device,
    )

    # ── DDM_W precompute (3D BAE run_039) ─────────────────────────────────────
    print("Precomputing GT for DDM_W...")
    scaler_params_w = scaler(pms_w) if pms_w is not None else None
    scaler_aoas_w   = scaler(ams_w) if ams_w is not None else None
    gt_coords_w, gt_aoas_w, gt_pressures_w, gt_w_w, gt_z_w, w_inits, params_w, case_nums = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_3d, lvae_model,
            ddm_w.w_mean, ddm_w.w_std,
            scaler_params_w, scaler_aoas_w, device,
        )

    # ── Generate: DDM_PCA ────────────────────────────────────────────────────
    print(f"\nRunning DDM_PCA ({args.n_passes} passes)...")
    all_coords_pca, all_aoas_pca, all_z_pca = [], [], []
    for i in range(args.n_passes):
        c, a, z = ddm_pca.generate(
            z_init=z_inits_pca.to(device),
            params=params_pca.to(device),
            device=device, T=args.T,
        )
        all_coords_pca.append(c)
        all_aoas_pca.append(a)
        all_z_pca.append(z)
        print(f"  PCA pass {i+1}/{args.n_passes}")
    gen_coords_pca = torch.stack(all_coords_pca).mean(0)
    gen_aoas_pca   = torch.stack(all_aoas_pca).mean(0)
    gen_z_pca      = torch.stack(all_z_pca).mean(0)

    # ── Generate: DDM_W ──────────────────────────────────────────────────────
    print(f"\nRunning DDM_W ({args.n_passes} passes)...")
    all_coords_w, all_aoas_w, all_pres_w, all_w_w = [], [], [], []
    for i in range(args.n_passes):
        c, a, p, _, w, _ = ddm_w.generate(
            w_init=w_inits.to(device),
            params=params_w.to(device),
            device=device, T=args.T,
        )
        all_coords_w.append(c)
        all_aoas_w.append(a)
        all_pres_w.append(p)
        all_w_w.append(w)
        print(f"  DDM_W pass {i+1}/{args.n_passes}")
    gen_coords_w = torch.stack(all_coords_w).mean(0)
    gen_aoas_w   = torch.stack(all_aoas_w).mean(0)
    gen_pres_w   = torch.stack(all_pres_w).mean(0)
    gen_w_w      = torch.stack(all_w_w).mean(0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics_pca = compute_metrics_pca(
        gen_coords_pca, gt_coords_pca, gen_aoas_pca, gt_aoas_pca,
        gen_z=gen_z_pca, gt_z=gt_z_pca,
    )
    metrics_w = compute_metrics_w(
        gen_coords_w, gt_coords_w, gen_aoas_w, gt_aoas_w,
        gen_pressures=gen_pres_w, gt_pressures=gt_pressures_w,
        gen_w=gen_w_w, gt_w=gt_w_w,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    all_keys = sorted(set(metrics_pca) | set(metrics_w))
    col_w = max(len(k) for k in all_keys) + 2

    print("\n" + "=" * 60)
    print(f"{'Metric':<{col_w}}  {'DDM_PCA':>12}  {'DDM_W':>12}")
    print("-" * 60)
    for k in all_keys:
        pca_val = f"{metrics_pca[k]:.6f}" if k in metrics_pca else "     —"
        w_val   = f"{metrics_w[k]:.6f}"   if k in metrics_w   else "     —"
        print(f"{k:<{col_w}}  {pca_val:>12}  {w_val:>12}")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"compare_pca_vs_ddm_w_{ts}")

    result = {
        "ddm_pca": {**metrics_pca, "checkpoint": args.ddm_pca_checkpoint,
                    "n_passes": args.n_passes, "n_test": len(test_dataset)},
        "ddm_w":   {**metrics_w,   "checkpoint": args.ddm_w_checkpoint,
                    "n_passes": args.n_passes, "n_test": len(test_dataset)},
    }
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)

    with open(base + ".txt", "w") as f:
        f.write(f"DDM_PCA checkpoint : {args.ddm_pca_checkpoint}\n")
        f.write(f"DDM_W   checkpoint : {args.ddm_w_checkpoint}\n")
        f.write(f"n_passes           : {args.n_passes}\n")
        f.write(f"n_test             : {len(test_dataset)}\n\n")
        f.write(f"{'Metric':<{col_w}}  {'DDM_PCA':>12}  {'DDM_W':>12}\n")
        f.write("-" * 60 + "\n")
        for k in all_keys:
            pca_val = f"{metrics_pca[k]:.6f}" if k in metrics_pca else "     —"
            w_val   = f"{metrics_w[k]:.6f}"   if k in metrics_w   else "     —"
            f.write(f"{k:<{col_w}}  {pca_val:>12}  {w_val:>12}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    shared_keys = [k for k in ["shape_mse", "aoa_mse", "mmd", "vendi"] if k in metrics_pca and k in metrics_w]
    x     = np.arange(len(shared_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_pca = ax.bar(x - width/2, [metrics_pca[k] for k in shared_keys], width,
                      label="DDM_PCA", color="steelblue")
    bars_w   = ax.bar(x + width/2, [metrics_w[k]   for k in shared_keys], width,
                      label="DDM_W (LVAE)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(shared_keys, rotation=15, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("DDM_PCA vs DDM_W: shared metrics")
    ax.legend()
    ax.bar_label(bars_pca, fmt="%.4f", fontsize=7, padding=2)
    ax.bar_label(bars_w,   fmt="%.4f", fontsize=7, padding=2)
    plt.tight_layout()
    fig.savefig(base + "_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved to {base}.json / .txt / _bar.png")


if __name__ == "__main__":
    main()
