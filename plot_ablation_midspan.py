"""
Plot one mid-span airfoil slice (GT blue vs Generated coral) per ablation
training size, all in a single figure.  Re-rendered from raw coordinates so
the output is fully sharp.

Usage
-----
    python plot_ablation_midspan.py
    python plot_ablation_midspan.py \
        --ablation_dir results/ddm_w_3d_ablation_v29 \
        --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
        --span_idx 7 \
        --wing_idx 0 \
        --out results/ablation_v29_midspan.pdf
"""

import argparse
import os
import re
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import (
    precompute_test_3d, load_bae_3d, load_lvae_3d, build_sampler,
)
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

COL_GT    = "#2166ac"
COL_GEN   = "#d6604d"

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":     150,
})


def find_checkpoints(ablation_dir):
    """Return sorted list of (n_samples, checkpoint_path) using seed=0 _best checkpoints."""
    entries = []
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s0"))):
        m = re.search(r"n(\d+)_s0", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        ckpts = sorted(glob.glob(os.path.join(n_dir, "*_best.pth")))
        if ckpts:
            entries.append((n, ckpts[-1]))
    return sorted(entries, key=lambda x: x[0])


def load_ddm(checkpoint, cfg, bae_model, lvae_model, device):
    ckpt    = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)
    w_dim   = cfg.lae_latent_dim
    saved   = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)

    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")
    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    return ddm_w, pms, ams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir",   default="results/ddm_w_3d_ablation_v29")
    p.add_argument("--bae_checkpoint", default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--lvae_checkpoint",default="results/lvae_3d/lvae_3d_v29_best.pth")
    p.add_argument("--span_idx", type=int, default=7,
                   help="Span slice index to plot (default 7 = mid-span for 15 slices)")
    p.add_argument("--wing_idx", type=int, default=0,
                   help="Which test wing to show (default 0)")
    p.add_argument("--out", default="results/ablation_v29_midspan.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    entries = find_checkpoints(args.ablation_dir)
    if not entries:
        print(f"No checkpoints found in {args.ablation_dir}")
        return
    print(f"Found {len(entries)} checkpoints: {[n for n,_ in entries]}")

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device          = str(device)

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    # Load dataset once — same GT across all checkpoints
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    n_cols = len(entries)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.4, 2.8), sharey=True)
    if n_cols == 1:
        axes = [axes]

    gt_coords_cached = None

    for col_idx, (n_samples, ckpt_path) in enumerate(entries):
        print(f"\n── n={n_samples}: {ckpt_path}")
        ddm_w, pms, ams = load_ddm(ckpt_path, cfg, bae_model, lvae_model, device)

        scaler_params = scaler(pms) if pms is not None else None
        scaler_aoas   = scaler(ams) if ams is not None else None

        gt_coords, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std, scaler_params, scaler_aoas, device,
        )

        if gt_coords_cached is None:
            gt_coords_cached = gt_coords  # identical across checkpoints

        with torch.no_grad():
            gen_coords, _, _, _, _, _ = ddm_w.generate(
                w_init=w_inits.to(device),
                params=params_norm.to(device),
                device=device,
            )

        # gt_coords / gen_coords: [N, S, 2, 192]  (2 = x/y, 192 points)
        s = args.span_idx
        wi = args.wing_idx
        gt_xy  = gt_coords_cached[wi, s].numpy()   # [2, 192]
        gen_xy = gen_coords[wi, s].cpu().numpy()    # [2, 192]

        ax = axes[col_idx]
        ax.plot(gt_xy[0],  gt_xy[1],  color=COL_GT,  lw=1.2, label="GT")
        ax.plot(gen_xy[0], gen_xy[1], color=COL_GEN, lw=1.2, ls="--", label="Gen")
        ax.set_title(f"n={n_samples}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(left=False, labelleft=(col_idx == 0))
        if col_idx == 0:
            ax.set_ylabel("y/c")
        ax.set_xlabel("x/c")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"DDM_W3D Ablation — mid-span slice (span idx {args.span_idx}, wing {args.wing_idx})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.splitext(args.out)[0] + "." + ext
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
