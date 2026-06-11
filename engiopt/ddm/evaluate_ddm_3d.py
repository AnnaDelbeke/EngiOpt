"""
Evaluation script for DDM-3D (direct BAE latent diffusion, no LVAE).

Usage
-----
    python -m engiopt.ddm.evaluate_ddm_3d \
        --checkpoint     results/ddm_3d/ddm_3d_v1_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        [--n_passes 10] [--seed 0]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.ddm.train_ddm_3d import DDM3D, load_bae_3d, precompute_z
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm import samplers
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]


def gaussian_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def compute_mmd(generated, real, gamma):
    n, m = generated.shape[0], real.shape[0]
    Kxx = gaussian_kernel(generated, generated, gamma)
    Kyy = gaussian_kernel(real,      real,      gamma)
    Kxy = gaussian_kernel(generated, real,      gamma)
    return (Kxx.sum()/(n*n) - 2*Kxy.sum()/(n*m) + Kyy.sum()/(m*m)).item()


def compute_vendi(samples, gamma):
    valid = torch.isfinite(samples).all(dim=-1)
    samples = samples[valid]
    if samples.shape[0] < 2:
        return float("nan")
    K = gaussian_kernel(samples, samples, gamma) / samples.shape[0]
    K = K + 1e-4 * torch.eye(K.shape[0], device=K.device)
    try:
        ev = torch.linalg.eigvalsh(K).clamp(min=1e-10)
    except torch._C._LinAlgError:
        K2 = K.double().cpu(); K2 = (K2 + K2.T) / 2
        ev = torch.linalg.eigvalsh(K2).clamp(min=1e-10).to(samples.device).float()
    ev = ev / ev.sum()
    return (-(ev * ev.log()).sum()).exp().item()


def compute_metrics(gen_coords, gt_coords, gen_aoas, gt_aoas,
                    gen_z=None, gt_z=None):
    n_slices  = gen_coords.shape[1]
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(n_slices):
        gen_s   = gen_coords[:, s]
        gt_s    = gt_coords[:, s]
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()
        gen_flat = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat  = gt_s.reshape(gt_s.shape[0],  -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
        vendi_gen_vals.append(float(np.nanmean([compute_vendi(gen_flat, g) for g in GAMMAS])))
        vendi_gt_vals.append(float(np.nanmean([compute_vendi(gt_flat,  g) for g in GAMMAS])))

    shape_mse /= n_slices
    mmd        = float(np.mean(mmd_vals))
    vendi_gt   = float(np.mean(vendi_gt_vals))
    vendi_norm = float(np.mean(vendi_gen_vals)) / vendi_gt if vendi_gt > 0 else 0.0
    aoa_mse    = ((gen_aoas - gt_aoas) ** 2).mean().item()

    out = {"shape_mse": shape_mse, "aoa_mse": aoa_mse, "mmd": mmd, "vendi": vendi_norm}
    if gen_z is not None and gt_z is not None:
        out["z_bae_mse"] = ((gen_z - gt_z) ** 2).mean().item()
        out["mmd_z"] = float(np.mean([compute_mmd(gen_z, gt_z, g) for g in GAMMAS]))
    return out


@torch.no_grad()
def generate(model: DDM3D, z_inits: torch.Tensor, params: torch.Tensor,
             device: str, T: int = None):
    """Denoise from noise → z_bae → BAE decode → coords + aoas."""
    B = z_inits.shape[0]
    schedule = model.sampler.schedule_x
    T = T or model.sampler.T

    z_noisy   = torch.randn(B, model.z_dim, device=device)
    aoa_noisy = torch.randn(B, 1, device=device)
    z_inits   = z_inits.to(device)
    params    = params.to(device)

    for i in reversed(range(T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        z_pred, aoa_pred = model.denoiser(z_noisy, aoa_noisy, params, z_inits, t)

        alpha_t      = schedule.alphas[i]
        alpha_bar_t  = schedule.alphas_cumprod[i]
        alpha_bar_t1 = schedule.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)
        beta_t       = 1.0 - alpha_t

        z0_est  = (z_noisy   - (1 - alpha_bar_t).sqrt() * z_pred)   / alpha_bar_t.sqrt().clamp(min=1e-8)
        aoa0_est = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
        z0_est   = z0_est.clamp(-5, 5)
        aoa0_est = aoa0_est.clamp(-5, 5)

        if i > 0:
            post_var  = beta_t * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8)
            z_mean    = (alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * z0_est
                         + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * z_noisy)
            aoa_mean  = (alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * aoa0_est
                         + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * aoa_noisy)
            z_noisy   = z_mean   + post_var.sqrt() * torch.randn_like(z_noisy)
            aoa_noisy = aoa_mean + post_var.sqrt() * torch.randn_like(aoa_noisy)
        else:
            z_noisy   = z0_est
            aoa_noisy = aoa0_est

    # Denormalise z
    if model.z_mean is not None:
        z_raw = z_noisy * model.z_std.to(device) + model.z_mean.to(device)
    else:
        z_raw = z_noisy

    coords = model.bae_model.decode(z_raw).cpu()  # [B, S, 2, 192]

    if model.scaler_aoas is not None:
        aoa_out = model.scaler_aoas.inverse_transform(aoa_noisy.cpu())
    else:
        aoa_out = aoa_noisy.cpu()

    return coords, aoa_out, z_raw.cpu()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--bae_checkpoint", type=str, required=True)
    p.add_argument("--n_passes", type=int, default=N_FORWARD_PASSES)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--T",        type=int, default=None)
    p.add_argument("--out_dir",  type=str, default="results/ddm_3d_evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    bae_model = load_bae_3d(args.bae_checkpoint, device, n_spans=15)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    z_dim = ckpt.get("z_dim", 64)
    pms   = ckpt.get("params_mean_std")
    ams   = ckpt.get("aoas_mean_std")

    sampler = samplers.BaselineSampler_AoA_3D(
        1000, start_x=1e-4, end_x=0.02, start_alpha=1e-4, end_alpha=0.02,
    )

    # infer dropout from checkpoint: spacing of 3 between linear layers means dropout was used
    net_keys = [k for k in ckpt["denoiser"].keys() if k.startswith("net.") and k.endswith(".weight")]
    spacing = int(net_keys[1].split(".")[1]) - int(net_keys[0].split(".")[1])
    dropout = 0.1 if spacing == 3 else 0.0
    denoiser = MLPDenoiser(w_dim=z_dim, c_dim=4, dropout=dropout)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.to(device).eval()

    model = DDM3D(
        denoiser=denoiser, bae_model=bae_model, sampler=sampler,
        z_dim=z_dim, c_dim=4, params_mean_std=pms, aoas_mean_std=ams,
        name="eval",
    )
    model.z_mean = ckpt.get("z_mean")
    model.z_std  = ckpt.get("z_std")
    print("DDM-3D loaded.")

    # Test dataset
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    # Pre-compute GT encodings
    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas_  = scaler(ams) if ams is not None else None

    z_gt, aoas_gt, params_all, z_inits = precompute_z(
        test_dataset, initial_by_case, bae_model, device
    )

    # GT coords via BAE decode
    with torch.no_grad():
        gt_coords = bae_model.decode(z_gt.to(device)).cpu()  # [N, S, 2, 192]

    # Normalise
    z_mean = model.z_mean
    z_std  = model.z_std
    z_inits_n = (z_inits - z_mean) / z_std if z_mean is not None else z_inits

    # Normalise params and aoas for conditioning
    params_n = torch.stack([
        scaler_params.transform(p) for p in params_all
    ]) if scaler_params is not None else params_all

    N = gt_coords.shape[0]
    print(f"Generating {args.n_passes} passes for {N} wings...")

    all_coords, all_aoas, all_z = [], [], []
    for pass_i in range(args.n_passes):
        coords_p, aoas_p, z_p = generate(model, z_inits_n, params_n, device, T=args.T)
        all_coords.append(coords_p)
        all_aoas.append(aoas_p)
        all_z.append(z_p)
        print(f"  Pass {pass_i+1}/{args.n_passes} done.")

    gen_coords = torch.stack(all_coords, dim=0).mean(0)  # [N, S, 2, 192]
    gen_aoas   = torch.stack(all_aoas,   dim=0).mean(0)  # [N, 1]
    gen_z      = torch.stack(all_z,      dim=0).mean(0)  # [N, z_dim]

    gt_aoas_t  = aoas_gt if aoas_gt.dim() == 1 else aoas_gt.squeeze(1)
    gen_aoas_t = gen_aoas.squeeze(1) if gen_aoas.dim() == 2 else gen_aoas

    # GT z (unnormalised BAE latents)
    gt_z = z_gt.to(gen_z.device)

    metrics = compute_metrics(gen_coords, gt_coords, gen_aoas_t, gt_aoas_t,
                              gen_z=gen_z, gt_z=gt_z)

    print("\n=== DDM-3D Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    base = os.path.join(args.out_dir, f"eval_{stem}_{ts}")

    metrics["checkpoint"] = args.checkpoint
    metrics["n_passes"]   = args.n_passes
    metrics["n_test"]     = N

    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(base + ".txt", "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"n_passes   : {args.n_passes}\n")
        f.write(f"n_test     : {N}\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")

    print(f"Saved to {base}.json / .txt")


if __name__ == "__main__":
    main()
