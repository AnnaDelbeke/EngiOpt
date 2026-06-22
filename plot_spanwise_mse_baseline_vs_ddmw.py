"""
Per-span shape MSE: 2D DDM reproduction (v7hope 250:1) vs DDM-W.

Loads the v7hope(250:1) checkpoint, generates on the Wings3D test split
(9 slices, same data used for training), computes shape MSE per span slice,
then overlays the DDM-W per-span shape MSE (15 slices, from
plot_spanwise_metrics_ddmw.py output).

Both curves are plotted against their actual eta (spanwise) positions so the
x-axis is comparable despite different slice counts.

Usage
-----
    .venv/bin/python plot_spanwise_mse_baseline_vs_ddmw.py \\
        [--ddmw_checkpoint  results/ddm_w_3d/ddm_w_3d_v29_best.pth] \\
        [--bae_checkpoint   results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt] \\
        [--lvae_checkpoint  results/lvae_3d/lvae_3d_v29_best.pth] \\
        [--n_passes 10] [--seed 0] \\
        [--out results/plots/spanwise_mse_baseline_vs_ddmw.pdf]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Wing_TL / engiopt imports ────────────────────────────────────────────────
from engibench.problems.wings3D.v0 import Wings3D
from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.train_ddm import precompute_latents, Config

# DDM-W pipeline
from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config as ConfigW, load_lvae_3d, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
_V7HOPE_CKPT = "results/ddm/ddm_v7hope(250:1).pth"
_BAE_2D_CKPT = "results/bezier_ae/run_006/models/bezier_ae_best.pt"

BLUE   = "#4C72B0"
ORANGE = "darkorange"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_2d_bae(checkpoint: str, device: str) -> BezierAutoencoder:
    cfg = Config()
    cfg.bae_checkpoint = checkpoint
    cfg.device = device
    model = BezierAutoencoder(
        n_control_points=cfg.n_control_points,
        n_data_points=cfg.n_data_points,
        batch_size=cfg.bae_batch_size,
        auto_batch=True,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Baseline (v7hope 250:1) per-span MSE ─────────────────────────────────────

@torch.no_grad()
def compute_baseline_spanwise_mse(args, device: str):
    print("Loading 2D BAE ...")
    bae_2d = load_2d_bae(_BAE_2D_CKPT, device)

    print("Loading v7hope(250:1) DDM checkpoint ...")
    ckpt = torch.load(_V7HOPE_CKPT, map_location="cpu", weights_only=False)

    ddm_model = DDM_AoAInit_3D(
        unet=ckpt["unet"].to(device),
        sampler=ckpt["sampler"],
        bae_model=bae_2d,
        params_mean_std=ckpt["params_mean_std"],
        aoas_mean_std=ckpt["aoas_mean_std"],
    )
    ddm_model.unet.eval()
    ddm_model.latent_mean = ckpt.get("latent_mean", 0.0)
    ddm_model.latent_std  = ckpt.get("latent_std",  1.0)
    print("v7hope DDM loaded.")

    print("Loading Wings3D test split (9 slices) ...")
    problem    = Wings3D(seed=args.seed)
    all_test   = list(problem.dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_final = [item for item in all_test if item["final"] == 1]
    print(f"  {len(test_final)} test wings")

    # eta positions are uniform across all cases — read from first item
    eta_9 = np.array(test_final[0]["transforms"], dtype=np.float32)  # [9]

    z_opts, gt_aoas, params_all, z_inits = precompute_latents(
        test_final, initial_by_case, bae_2d, device
    )
    N, S = z_opts.shape[:2]
    print(f"  Latents pre-computed: N={N}, S={S}")

    # Decode GT latents to get GT coords in the same BAE space
    print("  Decoding GT latents ...")
    gt_coords_list = []
    for i in range(N):
        slices = []
        for s in range(S):
            z_s = z_opts[i, s].unsqueeze(0).to(device)  # [1, 3, L]
            coords_s = bae_2d.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )[0].squeeze(0).cpu()  # [2, 192]
            slices.append(coords_s)
        gt_coords_list.append(torch.stack(slices))  # [S, 2, 192]
    gt_coords = torch.stack(gt_coords_list)  # [N, S, 2, 192]

    # Latent normalisation stats (model was trained on normalised latents)
    lat_mean = ddm_model.latent_mean
    lat_std  = ddm_model.latent_std
    if isinstance(lat_mean, torch.Tensor):
        lat_mean = lat_mean.to(device)
        lat_std  = lat_std.to(device)
        lat_mean_init = lat_mean[0]   # [3, 1] broadcast slice
        lat_std_init  = lat_std[0]
    else:
        lat_mean_init = lat_mean
        lat_std_init  = lat_std

    # Generate n_passes and average
    L = z_opts.shape[-1]
    print(f"  Generating ({args.n_passes} passes) ...")
    all_gen = []
    for pass_i in range(args.n_passes):
        gen_list = []
        for i in range(N):
            # Normalise z_init to match what the model saw during training
            z_init_raw = z_inits[i].unsqueeze(0).to(device)           # [1, 3, L]
            z_init_b   = (z_init_raw - lat_mean_init) / lat_std_init  # [1, 3, L]
            params_b   = ddm_model.scaler_params.transform(
                params_all[i].unsqueeze(0)
            ).to(device)                                               # [1, 4]
            noise_x     = torch.randn(1, S, 3, L, device=device)      # [1, S, 3, L]
            noise_alpha = torch.randn(1, 1,        device=device)
            gen_z, _ = ddm_model(
                [noise_x, noise_alpha], params_b, z_init_b, output_decoded=False
            )  # gen_z: [1, S, 3, L] in normalised latent space
            # Denormalise and clamp before decoding
            gen_z_raw = gen_z * lat_std + lat_mean
            gen_z_flat = gen_z_raw.squeeze(0)  # [S, 3, L]
            gen_z_flat[:, 0, :]  = gen_z_flat[:, 0, :].clamp(0.1, 2.0)
            gen_z_flat[:, 1:, :] = gen_z_flat[:, 1:, :].clamp(-2.228, 3.117)
            gen_af = ddm_model.bae_model.decode_z(
                gen_z_flat, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )[0]  # [S, 2, 192]
            gen_list.append(gen_af.cpu())
        all_gen.append(torch.stack(gen_list))  # [N, S, 2, 192]
        print(f"    pass {pass_i+1}/{args.n_passes}")

    gen_coords = torch.stack(all_gen, dim=0).mean(0)  # [N, S, 2, 192]

    # Per-span shape MSE (nanmean to tolerate any residual NaNs)
    diff_sq = (gen_coords - gt_coords) ** 2
    mse_per_span = torch.from_numpy(
        np.nanmean(diff_sq.numpy(), axis=(0, 2, 3))
    )
    return eta_9, mse_per_span.numpy()


# ── DDM-W per-span MSE ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ddmw_spanwise_mse(args, device: str):
    print("Loading DDM-W pipeline ...")
    cfg_w = ConfigW()
    cfg_w.bae_checkpoint  = args.bae_checkpoint
    cfg_w.lvae_checkpoint = args.lvae_checkpoint
    cfg_w.device          = device

    bae_3d  = load_bae_3d(args.bae_checkpoint, device)
    lvae    = load_lvae_3d(cfg_w, bae_3d)

    ckpt    = torch.load(args.ddmw_checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg_w)
    w_dim   = cfg_w.lae_latent_dim

    saved = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg_w.c_dim)
        denoiser.load_state_dict(saved)

    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")
    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae, bae_model=bae_3d,
        sampler=sampler, w_dim=w_dim, c_dim=cfg_w.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg_w.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.ddmw_checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    print("DDM-W loaded.")

    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"  {len(test_dataset)} test wings (15 slices)")

    gt_coords, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
        test_dataset, initial_by_case, bae_3d, lvae,
        ddm_w.w_mean, ddm_w.w_std, scaler_params, scaler_aoas, device,
    )

    eta_15 = np.array(test_dataset[0]["transforms"], dtype=np.float32)  # [15]

    all_coords = []
    for pass_i in range(args.n_passes):
        coords_p, _, _, _, _, _ = ddm_w.generate(
            w_init=w_inits.to(device),
            params=params_norm.to(device),
            device=device,
        )
        all_coords.append(coords_p.cpu())
        print(f"    pass {pass_i+1}/{args.n_passes}")

    gen_coords = torch.stack(all_coords, dim=0).mean(0)  # [N, S, 2, 192]
    mse_per_span = ((gen_coords - gt_coords) ** 2).mean(dim=(0, 2, 3)).numpy()  # [S]
    return eta_15, mse_per_span


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ddmw_checkpoint",  type=str,
                   default="results/ddm_w_3d/ddm_w_3d_v29_best.pth")
    p.add_argument("--bae_checkpoint",   type=str,
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--lvae_checkpoint",  type=str,
                   default="results/lvae_3d/lvae_3d_v29_best.pth")
    p.add_argument("--n_passes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out",      type=str,
                   default="results/plots/spanwise_mse_baseline_vs_ddmw.pdf")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eta_9,  mse_9  = compute_baseline_spanwise_mse(args, device)
    eta_15, mse_15 = compute_ddmw_spanwise_mse(args, device)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(eta_9,  mse_9,  color=ORANGE, marker="o", ms=4, lw=1.5,
            label="2D DDM (reproduced)")
    ax.plot(eta_15, mse_15, color=BLUE,   marker="s", ms=4, lw=1.5,
            label="DDM-W")

    ax.set_xlabel("Spanwise position $\\eta$ (m)", fontsize=10)
    ax.set_ylabel("Shape MSE", fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.6)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150, metadata={})
    if args.out.endswith(".pdf"):
        fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.out}")

    print("\n=== Per-span MSE summary ===")
    print("2D DDM (v7hope 250:1):")
    for eta, mse in zip(eta_9, mse_9):
        print(f"  eta={eta:.3f}  MSE={mse:.6f}")
    print("DDM-W:")
    for eta, mse in zip(eta_15, mse_15):
        print(f"  eta={eta:.3f}  MSE={mse:.6f}")


if __name__ == "__main__":
    main()
