"""
Evaluation script for DDM_PCA_3D checkpoints.

Usage
-----
    python -m engiopt.ddm.ddm_pca.evaluate_ddm_pca_3d \
        --checkpoint     results/ddm_pca_3d/ddm_pca_3d_v2_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        [--n_passes 10] [--seed 0] [--out_dir results/evaluation]
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.ddm.ddm_pca.ddm_pca_3d import DDM_PCA_3D, MLPDenoiser
from engiopt.ddm.ddm_pca.train_ddm_pca_3d import (
    Config, build_sampler, precompute_bae_latents_3d,
)
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

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


def compute_metrics(generated, gt_airfoils, gen_aoas, gt_aoas, gen_z=None, gt_z=None):
    n_slices  = generated.shape[1]
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(n_slices):
        gen_s = generated[:, s]
        gt_s  = gt_airfoils[:, s]
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
        out["mmd_z"] = float(np.mean([compute_mmd(gen_z, gt_z, g) for g in GAMMAS]))
    return out


def precompute_test_3d(test_dataset, initial_by_case, bae_model, pca,
                       z_mean, z_std, scaler_params, scaler_aoas, device):
    """Encode test wings through 3D BAE → PCA, return normalised z_inits and GT coords."""
    gt_coords_list = []
    gt_aoas_list   = []
    gt_z_list      = []
    z_inits_list   = []
    params_list    = []

    bae_model.eval()
    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            x_wing  = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)  # [1, S, 2, 192]
            z_bae   = bae_model.encode(x_wing)                            # [1, bae_latent_dim]
            gt_recon = bae_model.decode(z_bae).squeeze(0).cpu()           # [S, 2, 192]
            gt_coords_list.append(gt_recon)
            gt_aoas_list.append(torch.tensor(float(item["alpha"])))

            # GT PCA latent
            z_pca = torch.tensor(pca.transform(z_bae.cpu().numpy()), dtype=torch.float32).squeeze(0)
            gt_z_list.append(z_pca)

            # Initial wing PCA latent
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init = initial_by_case[case_num]
                ic   = torch.tensor(init["coords"],    dtype=torch.float32)
                it   = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ic_c = ic.clone()
                ic_c[:, :, 1] -= it.unsqueeze(1)
                te_x_i  = ic_c[:, 0, 0]
                ic_c[:, :, 0] += (1.0 - te_x_i).unsqueeze(1)
                le_x_i  = ic_c[:, :, 0].min(dim=1).values
                chord_i = 1.0 - le_x_i
                ic_c[:, :, 0] = (ic_c[:, :, 0] - le_x_i.unsqueeze(1)) / chord_i.unsqueeze(1)
                x_init  = ic_c.permute(0, 2, 1).unsqueeze(0).to(device)
                z_init  = bae_model.encode(x_init).cpu()
            else:
                z_init = z_bae.cpu()

            z_init_pca  = torch.tensor(pca.transform(z_init.numpy()), dtype=torch.float32).squeeze(0)
            z_init_norm = (z_init_pca - z_mean.squeeze(0)) / z_std.squeeze(0)
            z_inits_list.append(z_init_norm)

            flow   = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            params = scaler_params.transform(torch.tensor(flow, dtype=torch.float32))
            params_list.append(params)

    return (
        torch.stack(gt_coords_list),
        torch.stack(gt_aoas_list),
        torch.stack(gt_z_list),
        torch.stack(z_inits_list),
        torch.stack(params_list),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--bae_checkpoint", type=str, required=True)
    p.add_argument("--n_passes",  type=int, default=N_FORWARD_PASSES)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--T",         type=int, default=None)
    p.add_argument("--out_dir",   type=str, default="results/evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    bae_model = load_bae_3d(args.bae_checkpoint, device)

    ckpt     = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pca_path = args.checkpoint.replace("_best.pth", "_pca.pkl").replace(".pth", "_pca.pkl")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    cfg   = Config()
    z_dim = ckpt.get("z_dim", cfg.n_components)
    pms   = ckpt.get("params_mean_std")
    ams   = ckpt.get("aoas_mean_std")

    denoiser = MLPDenoiser(z_dim=z_dim, c_dim=cfg.c_dim).to(device)
    model = DDM_PCA_3D(
        denoiser=denoiser, pca=pca, bae_model=bae_model,
        sampler=build_sampler(cfg), z_dim=z_dim, c_dim=cfg.c_dim,
        w_aoa=ckpt.get("w_aoa", 1.0),
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    model.load(args.checkpoint, train_mode=False)
    model.denoiser.to(device)

    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    gt_coords, gt_aoas, gt_z, z_inits, params_norm = precompute_test_3d(
        test_dataset, initial_by_case, bae_model, pca,
        model.z_mean, model.z_std,
        model.scaler_params, model.scaler_aoas, device,
    )
    print(f"Pre-computed GT for {gt_coords.shape[0]} test wings.")

    all_coords, all_aoas, all_z = [], [], []
    for i in range(args.n_passes):
        c, a, z = model.generate(
            z_init=z_inits.to(device),
            params=params_norm.to(device),
            device=device, T=args.T,
        )
        all_coords.append(c)
        all_aoas.append(a)
        all_z.append(z)
        print(f"  Pass {i+1}/{args.n_passes} done.")

    gen_coords = torch.stack(all_coords).mean(0)
    gen_aoas   = torch.stack(all_aoas).mean(0)
    gen_z      = torch.stack(all_z).mean(0)

    metrics = compute_metrics(gen_coords, gt_coords, gen_aoas, gt_aoas,
                              gen_z=gen_z, gt_z=gt_z)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    base = os.path.join(args.out_dir, f"eval_{stem}_{ts}")

    metrics["checkpoint"] = args.checkpoint
    metrics["n_passes"]   = args.n_passes
    metrics["n_test"]     = len(test_dataset)

    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(base + ".txt", "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"n_passes   : {args.n_passes}\n")
        f.write(f"n_test     : {len(test_dataset)}\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")

    print(f"Saved to {base}.json / .txt")


if __name__ == "__main__":
    main()
