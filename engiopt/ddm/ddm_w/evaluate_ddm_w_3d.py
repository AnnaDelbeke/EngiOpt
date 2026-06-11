"""
Evaluation script for DDM_W3D checkpoints.

Usage
-----
    python -m engiopt.ddm.ddm_w.evaluate_ddm_w_3d \
        --checkpoint      results/ddm_w_3d/ddm_w_3d_v1_best.pth \
        --bae_checkpoint  results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v5_best.pth \
        [--n_passes 10] [--seed 0]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.lvae import plotting as lvae_plotting
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def shoelace_area(coords):
    """Cross-sectional area via shoelace formula.

    coords : [..., 2, N]  (x row, y row)
    returns: [...] scalar area per airfoil
    """
    x = coords[..., 0, :]   # [..., N]
    y = coords[..., 1, :]   # [..., N]
    # shoelace: 0.5 * |sum(x_i*(y_{i+1} - y_{i-1}))|
    area = 0.5 * torch.abs(
        (x * (torch.roll(y, -1, dims=-1) - torch.roll(y, 1, dims=-1))).sum(dim=-1)
    )
    return area


def compute_volume_metrics(gen_coords, gt_coords, area_case_ratios=None):
    """
    gen_coords      : [N, S, 2, 192]
    gt_coords       : [N, S, 2, 192]
    area_case_ratios: [N]  constraint thresholds (vol_con >= threshold required)

    Returns dict with:
      volume_mse          — MSE of per-wing integrated volume proxy (mean slice area)
      volume_constraint_sat — fraction of generated wings satisfying vol >= threshold * gt_vol
    """
    # mean cross-sectional area across spans as volume proxy
    gen_area = shoelace_area(gen_coords).mean(dim=1)   # [N]
    gt_area  = shoelace_area(gt_coords).mean(dim=1)    # [N]

    volume_mse = ((gen_area - gt_area) ** 2).mean().item()
    out = {"volume_mse": volume_mse}

    if area_case_ratios is not None:
        thresholds = area_case_ratios * gt_area   # [N]  minimum acceptable gen volume
        sat = (gen_area >= thresholds).float().mean().item()
        out["volume_constraint_sat"] = sat

    return out


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


def compute_metrics(generated, gt_airfoils, gen_aoas, gt_aoas,
                    gen_pressures=None, gt_pressures=None,
                    gen_w=None, gt_w=None,
                    gen_z=None, gt_z=None):
    n_slices  = generated.shape[1]
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(n_slices):
        gen_s    = generated[:, s]
        gt_s     = gt_airfoils[:, s]
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()
        gen_flat  = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat   = gt_s.reshape(gt_s.shape[0],  -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
        vendi_gen_vals.append(float(np.nanmean([compute_vendi(gen_flat, g) for g in GAMMAS])))
        vendi_gt_vals.append(float(np.nanmean([compute_vendi(gt_flat,  g) for g in GAMMAS])))

    shape_mse /= n_slices
    mmd        = float(np.mean(mmd_vals))
    vendi_gt   = float(np.mean(vendi_gt_vals))
    vendi_norm = float(np.mean(vendi_gen_vals)) / vendi_gt if vendi_gt > 0 else 0.0

    aoa_mse = ((gen_aoas - gt_aoas) ** 2).mean().item()

    out = {"shape_mse": shape_mse, "aoa_mse": aoa_mse, "mmd": mmd, "vendi": vendi_norm}
    if gen_pressures is not None and gt_pressures is not None:
        out["pressure_mse"] = ((gen_pressures - gt_pressures) ** 2).mean().item()
    if gen_w is not None and gt_w is not None:
        out["mmd_w"] = float(np.mean([compute_mmd(gen_w, gt_w, g) for g in GAMMAS]))
    if gen_z is not None and gt_z is not None:
        out["z_bae_mse"] = ((gen_z - gt_z) ** 2).mean().item()
        out["mmd_z"] = float(np.mean([compute_mmd(gen_z, gt_z, g) for g in GAMMAS]))

    return out


# ---------------------------------------------------------------------------
# GT pre-computation (3D pipeline)
# ---------------------------------------------------------------------------

def precompute_test_3d(test_dataset, initial_by_case, bae_model, lvae_model,
                       w_mean, w_std, scaler_params, scaler_aoas, device):
    """
    Encode GT test wings through 3D BAE + LVAE3D encoder.

    Returns
    -------
    gt_coords    : [N, S, 2, 192]   — BAE reconstruct of GT (normalised frame)
    gt_aoas      : [N]
    gt_pressures : [N, S, 192]
    gt_w         : [N, w_dim]
    w_inits_norm : [N, w_dim]
    params_norm  : [N, c_dim]
    case_nums    : list[int]
    """
    gt_coords_list    = []
    gt_aoas_list      = []
    gt_pressures_list = []
    gt_w_list         = []
    gt_z_list         = []
    w_inits_list      = []
    params_list       = []
    case_nums_list    = []

    bae_model.eval()
    lvae_model.encoder.eval()

    lvae_ps = getattr(lvae_model, 'scaler_params', None)

    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],        dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"],     dtype=torch.float32)  # [S]
            pressure  = torch.tensor(item["coef_pressure"], dtype=torch.float32)

            # Normalise: TE y-shift + TE x → 1.0 + LE x → 0.0
            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            x_wing   = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)  # [1, S, 2, 192]
            z_bae    = bae_model.encode(x_wing)                            # [1, bae_latent_dim]
            gt_recon = bae_model.decode(z_bae).squeeze(0).cpu()            # [S, 2, 192]

            gt_coords_list.append(gt_recon)
            gt_pressures_list.append(pressure)
            gt_aoas_list.append(torch.tensor(float(item["alpha"])))
            gt_z_list.append(z_bae.squeeze(0).cpu())

            flow    = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            if lvae_ps is not None:
                flow_lvae_np = lvae_ps.transform(flow_np)
            else:
                flow_lvae_np = flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)

            # GT w
            from engiopt.lvae.train_lvae_3d import LAEEncoderJoint3D
            pressure_dev = pressure.unsqueeze(0).to(device)
            is_joint = isinstance(lvae_model.encoder, LAEEncoderJoint3D)
            if is_joint:
                w_gt = lvae_model.encoder(z_bae, pressure_dev, flow_lvae).squeeze(0).cpu()
            else:
                w_gt = lvae_model.encoder(z_bae, pressure_dev, flow_lvae).squeeze(0).cpu()
            gt_w_list.append(w_gt)

            # w_init
            case_num = int(item["case_num"])
            case_nums_list.append(case_num)
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
                x_init = ic_c.permute(0, 2, 1).unsqueeze(0).to(device)
                z_init = bae_model.encode(x_init)
                init_pressure = torch.tensor(init["coef_pressure"], dtype=torch.float32).unsqueeze(0).to(device)
            else:
                z_init = z_bae
                init_pressure = pressure_dev

            if is_joint:
                w_init = lvae_model.encoder(z_init, init_pressure, flow_lvae).squeeze(0).cpu()
            else:
                w_init = lvae_model.encoder(z_init, init_pressure, flow_lvae).squeeze(0).cpu()
            w_init_norm = (w_init - w_mean.squeeze(0)) / w_std.squeeze(0)
            w_inits_list.append(w_init_norm)

            flow_t      = torch.tensor(flow, dtype=torch.float32)
            params_norm = scaler_params.transform(flow_t)
            params_list.append(params_norm)

    return (
        torch.stack(gt_coords_list),    # [N, S, 2, 192]
        torch.stack(gt_aoas_list),      # [N]
        torch.stack(gt_pressures_list), # [N, S, 192]
        torch.stack(gt_w_list),         # [N, w_dim]
        torch.stack(gt_z_list),         # [N, bae_latent_dim]
        torch.stack(w_inits_list),      # [N, w_dim]
        torch.stack(params_list),       # [N, c_dim]
        case_nums_list,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
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

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    # Load DDM_W3D checkpoint
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
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    print("DDM_W3D loaded.")

    w_mean = ddm_w.w_mean
    w_std  = ddm_w.w_std

    # Scalers for condition normalisation (re-built from ckpt stats)
    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    area_case_ratios = torch.tensor(
        [float(item["area_case_ratio"]) for item in test_dataset], dtype=torch.float32
    )

    gt_coords, gt_aoas, gt_pressures, gt_w, gt_z, w_inits, params_norm, case_nums = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            w_mean, w_std, scaler_params, scaler_aoas, device,
        )
    N = gt_coords.shape[0]
    print(f"Pre-computed GT for {N} test wings.")

    all_coords, all_aoas, all_pressures, all_w, all_z = [], [], [], [], []
    for pass_i in range(args.n_passes):
        coords_p, aoas_p, pres_p, _, w_p, z_p = ddm_w.generate(
            w_init=w_inits.to(device),
            params=params_norm.to(device),
            device=device,
            T=args.T,
        )
        all_coords.append(coords_p)
        all_aoas.append(aoas_p)
        all_pressures.append(pres_p)
        all_w.append(w_p)
        all_z.append(z_p)
        print(f"  Pass {pass_i+1}/{args.n_passes} done.")

    gen_coords    = torch.stack(all_coords,    dim=0).mean(0)
    gen_aoas      = torch.stack(all_aoas,      dim=0).mean(0)
    gen_pressures = torch.stack(all_pressures, dim=0).mean(0)
    gen_z         = torch.stack(all_z,         dim=0).mean(0)

    # For MMD we want the full sample cloud (all passes), not the per-sample mean
    gen_w_all = torch.cat(all_w, dim=0)        # [N * n_passes, w_dim]
    gt_w_all  = gt_w.repeat(args.n_passes, 1)  # match cardinality

    metrics = compute_metrics(
        gen_coords, gt_coords, gen_aoas, gt_aoas,
        gen_pressures, gt_pressures,
        gen_w=gen_w_all, gt_w=gt_w_all,
        gen_z=gen_z, gt_z=gt_z,
    )
    metrics.update(compute_volume_metrics(gen_coords, gt_coords, area_case_ratios))

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    base = os.path.join(args.out_dir, f"eval_{stem}_{ts}")

    # ── Plots ────────────────────────────────────────────────────────────────
    flow_conditions = [
        {"mach": item["mach"], "reynolds": item["reynolds"],
         "cl_target": item["cl_target"], "aoa": float(item["alpha"])}
        for item in test_dataset
    ]

    lvae_plotting.plot_airfoil_cp_comparison(
        gen_coords, gt_coords,
        gen_pressures, gt_pressures,
        sample_indices=list(range(min(3, N))),
        slice_indices=list(range(gt_coords.shape[1])),
        save_path=base + "_airfoil_cp.png",
        pred_label="DDM_W3D generated",
        flow_conditions=flow_conditions,
    )

    lvae_plotting.plot_airfoil_slices_comparison(
        gen_coords, gt_coords,
        sample_indices=list(range(min(5, N))),
        slice_indices=[0, 4, 8, 11, 14],
        save_path=base + "_slice_comparison.png",
    )

    for si in range(min(3, N)):
        lvae_plotting.plot_wing3d_comparison(
            gen_coords, gt_coords,
            sample_idx=si,
            save_path=base + f"_wing3d_s{si}.png",
            title_a="GT",
            title_b="DDM_W3D generated",
        )

    lvae_plotting.plot_reconstruction_error_histograms(
        gen_coords, gt_coords,
        gen_pressures, gt_pressures,
        rec_perfs=None, gt_perfs=None,
        save_path=base + "_error_hist.png",
    )

    print(f"Plots saved to {args.out_dir}/")

    metrics["checkpoint"] = args.checkpoint
    metrics["n_passes"]   = args.n_passes
    metrics["n_test"]     = N
    metrics["case_nums"]  = case_nums

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
