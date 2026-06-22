"""
Spanwise metrics plot for DDM_W3D — replicates Fig. 17 style with an
additional pressure MSE panel.

Four panels (top to bottom):
  1. Shape MSE per span
  2. MMD per span
  3. Vendi score per span (generated vs ground-truth)
  4. Pressure MSE per span

Usage
-----
    .venv/bin/python plot_spanwise_metrics_ddmw.py \
        --checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
        --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
        [--n_passes 10] [--seed 0] [--out results/plots/spanwise_metrics_ddmw.pdf]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
GAMMAS       = [0.5, 25, 50, 100]
WING_LEN     = 2.55   # metres, tip η


# ---------------------------------------------------------------------------
# Metric helpers (per-span)
# ---------------------------------------------------------------------------

def gaussian_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def mmd_per_span(gen, gt):
    """gen, gt: [N, S, C] — returns [S] MMD averaged over GAMMAS."""
    S = gen.shape[1]
    vals = []
    for s in range(S):
        g = gen[:, s].reshape(gen.shape[0], -1)
        r = gt[:, s].reshape(gt.shape[0],  -1)
        n, m = g.shape[0], r.shape[0]
        span_mmds = []
        for gamma in GAMMAS:
            Kxx = gaussian_kernel(g, g, gamma)
            Kyy = gaussian_kernel(r, r, gamma)
            Kxy = gaussian_kernel(g, r, gamma)
            span_mmds.append((Kxx.sum()/(n*n) - 2*Kxy.sum()/(n*m) + Kyy.sum()/(m*m)).item())
        vals.append(float(np.mean(span_mmds)))
    return np.array(vals)


def vendi_per_span(samples):
    """samples: [N, S, C] — returns [S] Vendi scores."""
    S = samples.shape[1]
    vals = []
    for s in range(S):
        x = samples[:, s].reshape(samples.shape[0], -1)
        valid = torch.isfinite(x).all(dim=-1)
        x = x[valid]
        if x.shape[0] < 2:
            vals.append(float("nan"))
            continue
        span_vendis = []
        for gamma in GAMMAS:
            K  = gaussian_kernel(x, x, gamma) / x.shape[0]
            K  = K + 1e-4 * torch.eye(K.shape[0], device=K.device)
            ev = torch.linalg.eigvalsh(K).clamp(min=1e-10)
            ev = ev / ev.sum()
            span_vendis.append((-(ev * ev.log()).sum()).exp().item())
        vals.append(float(np.nanmean(span_vendis)))
    return np.array(vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--n_passes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out",      type=str, default="results/plots/spanwise_metrics_ddmw.pdf")
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    ckpt    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)
    w_dim   = cfg.lae_latent_dim

    saved = ckpt["denoiser"]
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
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    print("DDM_W3D loaded.")

    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    gt_coords, gt_aoas, gt_pressures, gt_w, gt_z, w_inits, params_norm, _ = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std, scaler_params, scaler_aoas, device,
        )

    # Generate (average over n_passes)
    all_coords, all_pressures = [], []
    for i in range(args.n_passes):
        coords_p, _, pres_p, _, _, _ = ddm_w.generate(
            w_init=w_inits.to(device),
            params=params_norm.to(device),
            device=device,
        )
        all_coords.append(coords_p.cpu())
        all_pressures.append(pres_p.cpu())
        print(f"  Pass {i+1}/{args.n_passes}")

    gen_coords    = torch.stack(all_coords,    dim=0).mean(0)   # [N, S, 2, 192]
    gen_pressures = torch.stack(all_pressures, dim=0).mean(0)   # [N, S, 192]

    S   = gen_coords.shape[1]
    eta = np.linspace(0, WING_LEN, S)

    # ── Per-span metrics ─────────────────────────────────────────────────────

    # Shape MSE [S]
    shape_mse_per_span = ((gen_coords - gt_coords) ** 2).mean(dim=(0, 2, 3)).numpy()

    # MMD [S]  (flatten 2×192 → 384 per slice)
    mmd_vals = mmd_per_span(
        gen_coords.reshape(*gen_coords.shape[:2], -1),
        gt_coords.reshape(*gt_coords.shape[:2],  -1),
    )

    # Vendi [S]
    vendi_gen = vendi_per_span(gen_coords.reshape(*gen_coords.shape[:2], -1))
    vendi_gt  = vendi_per_span(gt_coords.reshape(*gt_coords.shape[:2],  -1))

    # Pressure MSE [S]
    pressure_mse_per_span = ((gen_pressures - gt_pressures) ** 2).mean(dim=(0, 2)).numpy()

    # ── Plot ─────────────────────────────────────────────────────────────────
    BLUE   = "#4C72B0"
    GRAY   = "#888888"
    GREEN  = "#55A868"
    span_labels = [str(i + 1) for i in range(S)]

    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Panel 1 — Shape MSE
    ax = axes[0]
    ax.plot(eta, shape_mse_per_span, color=BLUE, marker="o", ms=4, lw=1.5)
    ax.set_ylabel("Shape MSE", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, lw=0.4, alpha=0.6)

    # Panel 2 — Pressure MSE
    ax = axes[1]
    ax.plot(eta, pressure_mse_per_span, color="darkorange", marker="o", ms=4, lw=1.5)
    ax.set_ylabel("Pressure MSE", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, lw=0.4, alpha=0.6)

    # Panel 3 — MMD
    ax = axes[2]
    ax.plot(eta, mmd_vals, color="tab:red", marker="o", ms=4, lw=1.5)
    ax.set_ylabel("MMD", fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.6)

    # Panel 4 — Vendi score
    ax = axes[3]
    ax.plot(eta, vendi_gen, color=GREEN, marker="o", ms=4, lw=1.5, label="Generated")
    ax.plot(eta, vendi_gt,  color=GRAY,  marker="s", ms=4, lw=1.5, linestyle="--", label="Ground Truth")
    ax.set_ylabel("Vendi Score", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, lw=0.4, alpha=0.6)

    # Shared x-axis: span numbers on every panel
    for ax in axes:
        ax.set_xticks(eta)
        ax.set_xticklabels(span_labels, fontsize=8)
    axes[-1].set_xlabel("Span slice", fontsize=10)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150, metadata={})
    # also save png alongside if pdf requested
    if args.out.endswith(".pdf"):
        fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
