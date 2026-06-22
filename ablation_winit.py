"""
Ablation: does w_init actually steer DDM-W generation?

Runs the DDM-W evaluation twice on the test set:
  1. Normal:   w_init = LVAE encoding of the initial wing  (baseline)
  2. Zeroed:   w_init = zero vector                        (ablation)

If metrics are similar, the denoiser ignores w_init.
If metrics degrade, w_init is doing real work.

Usage
-----
    python ablation_winit.py \
        --checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
        --bae_checkpoint  results/bezier_ae_3d/run_047/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
        [--n_passes 10] [--seed 0]
"""

import argparse
import torch
import numpy as np

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d, compute_metrics

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--n_passes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def run_passes(ddm_w, w_inits, params_norm, device, n_passes, seed, zero_winit=False):
    torch.manual_seed(seed)
    all_coords, all_aoas, all_pressures = [], [], []
    w_init_input = torch.zeros_like(w_inits) if zero_winit else w_inits
    for _ in range(n_passes):
        coords_p, aoas_p, pres_p, _, _, _ = ddm_w.generate(
            w_init=w_init_input.to(device),
            params=params_norm.to(device),
            device=device,
        )
        all_coords.append(coords_p)
        all_aoas.append(aoas_p)
        all_pressures.append(pres_p)
    gen_coords    = torch.stack(all_coords,    dim=0).mean(0)
    gen_aoas      = torch.stack(all_aoas,      dim=0).mean(0)
    gen_pressures = torch.stack(all_pressures, dim=0).mean(0) if all_pressures[0] is not None else None
    return gen_coords, gen_aoas, gen_pressures


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    ckpt    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)

    w_dim = cfg.lae_latent_dim
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
        name="ablation_winit",
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)

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
    print(f"GT pre-computed for {gt_coords.shape[0]} wings.")

    print("\nRunning NORMAL (w_init = real initial wing encoding)...")
    gen_coords_n, gen_aoas_n, gen_pressures_n = run_passes(
        ddm_w, w_inits, params_norm, device, args.n_passes, args.seed, zero_winit=False
    )
    metrics_normal = compute_metrics(
        gen_coords_n, gt_coords, gen_aoas_n, gt_aoas,
        gen_pressures_n, gt_pressures,
    )

    print("Running ZEROED (w_init = zero vector)...")
    gen_coords_z, gen_aoas_z, gen_pressures_z = run_passes(
        ddm_w, w_inits, params_norm, device, args.seed + 1, args.seed + 1, zero_winit=True
    )
    metrics_zeroed = compute_metrics(
        gen_coords_z, gt_coords, gen_aoas_z, gt_aoas,
        gen_pressures_z, gt_pressures,
    )

    print("\n" + "=" * 55)
    print(f"{'Metric':<20}  {'Normal':>12}  {'Zeroed w_init':>13}")
    print("-" * 55)
    all_keys = sorted(set(metrics_normal) | set(metrics_zeroed))
    for k in all_keys:
        v_n = metrics_normal.get(k, float("nan"))
        v_z = metrics_zeroed.get(k, float("nan"))
        print(f"  {k:<18}  {v_n:>12.6f}  {v_z:>13.6f}")
    print("=" * 55)
    print("\nInterpretation:")
    shape_delta = abs(metrics_zeroed.get("shape_mse", 0) - metrics_normal.get("shape_mse", 0))
    aoa_delta   = abs(metrics_zeroed.get("aoa_mse",   0) - metrics_normal.get("aoa_mse",   0))
    rel_shape   = shape_delta / max(metrics_normal.get("shape_mse", 1e-9), 1e-9)
    rel_aoa     = aoa_delta   / max(metrics_normal.get("aoa_mse",   1e-9), 1e-9)
    print(f"  Shape MSE relative change: {rel_shape*100:.1f}%")
    print(f"  AoA   MSE relative change: {rel_aoa*100:.1f}%")
    if rel_shape < 0.05 and rel_aoa < 0.05:
        print("  → w_init appears to have negligible effect (<5% on both metrics).")
    else:
        print("  → w_init has a meaningful effect on generation quality.")


if __name__ == "__main__":
    main()
