"""
Evaluate the Wing_TL DDM baseline checkpoint on the new_dataset test split.

Computes the same thesis metrics as evaluate_ddm_w_3d.py:
  shape_mse, aoa_mse, mmd, vendi, volume_mse, volume_constraint_sat

Usage (from EngiOpt root, venv activated):
    python eval_wing_tl_baseline.py [--n_passes 10] [--seed 0] [--out_dir results/evaluation]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import importlib
import importlib.abc
import importlib.machinery
import importlib.util

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

# Wing_TL source: insert its src directory, but stub out its broken __init__
# (wing_tl/__init__.py imports omegaconf which is not in this venv).
_WING_TL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wing_TL", "src")
sys.path.insert(0, _WING_TL_SRC)


class _StubLoader(importlib.abc.Loader):
    """Loader that executes nothing — gives wing_tl a no-op __init__."""
    def exec_module(self, mod):
        pass


class _WingTLInitStubber(importlib.abc.MetaPathFinder):
    """Intercept the wing_tl package import and replace __init__ with a no-op."""
    def find_spec(self, fullname, path, target=None):
        if fullname == "wing_tl":
            init_path = os.path.join(_WING_TL_SRC, "wing_tl", "__init__.py")
            spec = importlib.util.spec_from_file_location(
                "wing_tl",
                init_path,
                submodule_search_locations=[os.path.join(_WING_TL_SRC, "wing_tl")],
                loader=_StubLoader(),
            )
            return spec
        return None


sys.meta_path.insert(0, _WingTLInitStubber())

import wing_tl.bae.bezier_ae as _bae_mod
from wing_tl.ddm.ddm import DDM_AoAInit_3D
from wing_tl.ddm import unets, samplers
from wing_tl.data_processing.utils import scaler


def load_model_wandb(model_folder_name, model_name, model_folder_basepath, device="cpu"):
    """Wrapper around Wing_TL's load_model_wandb that adds map_location."""
    load_path = os.path.join(model_folder_basepath, model_folder_name, "files", model_name)
    model = torch.load(load_path, map_location=device, weights_only=False).to(device)
    print(f"Loaded BAE from {load_path}")
    model.eval()
    model.change_auto_batch(True)
    return model

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

# ── Paths ────────────────────────────────────────────────────────────────────
_WING_TL_ROOT    = os.path.join(os.path.dirname(__file__), "Wing_TL")
_DDM_CKPT        = os.path.join(_WING_TL_ROOT, "models", "ddm_3D", "ddm_baseline3D.pth")
_BAE_FOLDER      = "run-20250311_164414-pwl7ylee"
_BAE_NAME        = "bezier_ae"
_BAE_BASE        = os.path.join(_WING_TL_ROOT, "models", "bae")
_SLICES_PKL      = os.path.join(_WING_TL_ROOT, "data", "processed", "new_dataset_slices.pkl")
_SCALARS_PKL     = os.path.join(_WING_TL_ROOT, "data", "processed", "new_dataset_scalars.pkl")
_DDM_CFG_PATH    = os.path.join(_WING_TL_ROOT, "src", "wing_tl", "configs", "ddm_baseline3D.json")

# Subset of span indices used for evaluation (matches 9-slice Wing_TL convention)
_N_SLICES = 9
# Wing_TL BAE was trained on 2D slices: each slice has shape [2, 192]
_N_PTS    = 192

GAMMAS = [0.5, 25, 50, 100]


# ── Metric helpers ────────────────────────────────────────────────────────────

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


def shoelace_area(coords):
    """coords: [..., 2, N] → [...] area per airfoil."""
    x = coords[..., 0, :]
    y = coords[..., 1, :]
    return 0.5 * torch.abs(
        (x * (torch.roll(y, -1, dims=-1) - torch.roll(y, 1, dims=-1))).sum(dim=-1)
    )


def compute_metrics(gen_coords, gt_coords, gen_aoas, gt_aoas, area_case_ratios=None):
    """
    gen_coords : [N, S, 2, 192]
    gt_coords  : [N, S, 2, 192]
    gen_aoas   : [N]
    gt_aoas    : [N]
    """
    N, S = gen_coords.shape[:2]
    shape_mse_total = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(S):
        gen_s = gen_coords[:, s]          # [N, 2, 192]
        gt_s  = gt_coords[:, s]
        shape_mse_total += ((gen_s - gt_s) ** 2).mean().item()

        gen_flat = gen_s.reshape(N, -1)
        gt_flat  = gt_s.reshape(N, -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
        vendi_gen_vals.append(float(np.nanmean([compute_vendi(gen_flat, g) for g in GAMMAS])))
        vendi_gt_vals.append(float(np.nanmean([compute_vendi(gt_flat,  g) for g in GAMMAS])))

    shape_mse = shape_mse_total / S
    mmd       = float(np.mean(mmd_vals))
    vendi_gt  = float(np.mean(vendi_gt_vals))
    vendi_norm = float(np.mean(vendi_gen_vals)) / vendi_gt if vendi_gt > 0 else 0.0

    aoa_mse = ((gen_aoas - gt_aoas) ** 2).mean().item()

    out = {
        "shape_mse": shape_mse,
        "aoa_mse":   aoa_mse,
        "mmd":       mmd,
        "vendi":     vendi_norm,
    }

    # Volume metrics
    gen_area = shoelace_area(gen_coords).mean(dim=1)  # [N]
    gt_area  = shoelace_area(gt_coords).mean(dim=1)
    out["volume_mse"] = ((gen_area - gt_area) ** 2).mean().item()
    if area_case_ratios is not None:
        thresholds = area_case_ratios * gt_area
        out["volume_constraint_sat"] = (gen_area >= thresholds).float().mean().item()

    return out


# ── Data preparation ──────────────────────────────────────────────────────────

def normalise_coords(coords_np):
    """
    Normalise a single slice's coordinates to match Wing_TL BAE convention:
      - TE y-shift removed  (TE is first point)
      - TE x set to 1.0
      - LE x set to 0.0

    coords_np : [192, 2]  (x, y columns)
    Returns   : [2, 192]  torch tensor (row 0 = x, row 1 = y)
    """
    c = coords_np.copy().astype(np.float32)
    te_y = c[0, 1]
    c[:, 1] -= te_y                   # remove TE y-shift
    te_x = c[0, 0]
    c[:, 0] += (1.0 - te_x)          # shift so TE x → 1.0
    le_x = c[:, 0].min()
    chord = 1.0 - le_x
    c[:, 0] = (c[:, 0] - le_x) / chord   # scale so LE x → 0.0
    return torch.tensor(c.T, dtype=torch.float32)  # [2, 192]


def build_test_tensors(test_items, bae, scaler_params, scaler_aoas, device):
    """
    Encode the test set through the Wing_TL BAE (slice by slice).

    Returns
    -------
    gt_coords_all  : [N, S, 2, 192]   BAE-reconstructed GT slices
    gt_aoas_all    : [N]
    encoded_init_all : [N, S, 3, 30]  BAE latent of initial wing (per slice)
    params_all     : [N, 4]           normalised flow conditions
    area_ratios    : [N]
    """
    gt_coords_list   = []
    gt_aoas_list     = []
    enc_init_list    = []
    params_list      = []
    area_ratio_list  = []

    # Separate initial items by case for lookup
    initial_by_case = {item["case_num"]: item for item in test_items if item["initial"] == 1}
    final_items     = [item for item in test_items if item["final"] == 1]

    bae.eval()
    with torch.no_grad():
        for item in final_items:
            coords_np = item["coords"]   # [S, 192, 2]
            S_item = coords_np.shape[0]

            # Use the first _N_SLICES slices (root → tip ordered by eta)
            n = min(S_item, _N_SLICES)

            # ── GT reconstruction through BAE ─────────────────────────────
            gt_slices = []
            for s in range(n):
                c_norm = normalise_coords(coords_np[s]).unsqueeze(0).to(device)  # [1, 2, 192]
                _, _, _, cp, w = bae(c_norm)
                z = torch.cat([w, cp], dim=1)        # [1, 3, 30]
                recon = bae.decode_z(z, z_ae_mode=True, denormalize_output=True, normalized_data=False)[0]
                gt_slices.append(recon.squeeze(0).cpu())   # [2, 192]
            gt_coords_list.append(torch.stack(gt_slices))  # [S, 2, 192]

            gt_aoas_list.append(torch.tensor(float(item["alpha"]), dtype=torch.float32))

            # ── Encode initial wing through BAE ───────────────────────────
            case_num = int(item["case_num"])
            init_item = initial_by_case.get(case_num, item)   # fallback to self
            init_coords_np = init_item["coords"]

            enc_init_slices = []
            for s in range(n):
                ic_norm = normalise_coords(init_coords_np[s]).unsqueeze(0).to(device)
                _, _, _, cp_i, w_i = bae(ic_norm)
                z_i = torch.cat([w_i, cp_i], dim=1)  # [1, 3, 30]
                enc_init_slices.append(z_i.squeeze(0).cpu())   # [3, 30]
            enc_init_list.append(torch.stack(enc_init_slices))  # [S, 3, 30]

            # ── Flow conditions ───────────────────────────────────────────
            flow = torch.tensor(
                [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            )
            params_norm = scaler_params.transform(flow.unsqueeze(0)).squeeze(0)
            params_list.append(params_norm)

            area_ratio_list.append(torch.tensor(float(item["area_case_ratio"]), dtype=torch.float32))

    gt_coords  = torch.stack(gt_coords_list)    # [N, S, 2, 192]
    gt_aoas    = torch.stack(gt_aoas_list)      # [N]
    enc_inits  = torch.stack(enc_init_list)     # [N, S, 3, 30]
    params_all = torch.stack(params_list)       # [N, 4]
    area_ratios = torch.stack(area_ratio_list)  # [N]

    return gt_coords, gt_aoas, enc_inits, params_all, area_ratios


# ── Generation ────────────────────────────────────────────────────────────────

def generate_wing(ddm_model, enc_init_slices, params_norm, sampler, scaler_aoas, device):
    """
    Generate all slices for one wing.

    enc_init_slices : [S, 3, 30]
    params_norm     : [4]

    Returns gen_coords [S, 2, 192], gen_aoa scalar (deg)
    """
    S = enc_init_slices.shape[0]
    gen_slices = []
    gen_aoa_sum = 0.0

    params_b = params_norm.unsqueeze(0).to(device)           # [1, 4]

    for s in range(S):
        enc_init_b = enc_init_slices[s].unsqueeze(0).to(device)  # [1, 3, 30]

        noise_x     = torch.randn(1, 3, 30, device=device)
        noise_alpha = torch.randn(1, 1, device=device)

        gen_af, gen_aoa = ddm_model(
            [noise_x, noise_alpha],
            params_b,
            enc_init_b,
            output_decoded=True,
        )
        gen_slices.append(gen_af.squeeze(0).cpu())   # [2, 192]
        gen_aoa_sum += float(gen_aoa.squeeze().item())

    gen_coords = torch.stack(gen_slices)   # [S, 2, 192]
    gen_aoa    = gen_aoa_sum / S           # average over slices (all should be same)
    return gen_coords, gen_aoa


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_passes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out_dir",  type=str, default="results/evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load BAE ─────────────────────────────────────────────────────────────
    print("Loading Wing_TL BAE ...")
    bae = load_model_wandb(
        _BAE_FOLDER, _BAE_NAME,
        model_folder_basepath=_BAE_BASE,
        device=device,
    )
    bae.eval()

    # ── Load DDM ─────────────────────────────────────────────────────────────
    print("Loading Wing_TL DDM checkpoint ...")
    import json as _json
    with open(_DDM_CFG_PATH, "r") as f:
        config = _json.load(f)
    ddm_cfg     = config["ddm_cfg"]
    unet_cfg    = config["unet_cfg"]
    sampler_cfg = config["sampler_cfg"]

    ddm_sampler = samplers.BaselineSampler_AoA(**sampler_cfg)
    unet_obj    = unets.Unet_AoAInit3D(**unet_cfg).to(device)

    ddm_model = DDM_AoAInit_3D(
        unet_obj, ddm_sampler, bae,
        **ddm_cfg,
        checkpoint=_DDM_CKPT,
        train_mode=False,
    )
    ddm_model.unet.eval()

    sc_params = ddm_model.scaler_params
    sc_aoas   = ddm_model.scaler_aoas

    # ── Load test data ────────────────────────────────────────────────────────
    print("Loading test dataset ...")
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test    = list(new_dataset["test"])
    print(f"  {len(all_test)} raw test items ({sum(i['final']==1 for i in all_test)} final wings)")

    gt_coords, gt_aoas, enc_inits, params_all, area_ratios = build_test_tensors(
        all_test, bae, sc_params, sc_aoas, device
    )
    N = gt_coords.shape[0]
    S = gt_coords.shape[1]
    print(f"  GT tensors built: N={N}, S={S}")

    # ── Generate (n_passes forward passes, then average) ─────────────────────
    all_gen_coords = []
    all_gen_aoas   = []

    for pass_i in range(args.n_passes):
        gen_coords_list = []
        gen_aoas_list   = []
        with torch.no_grad():
            for i in range(N):
                gc, ga = generate_wing(
                    ddm_model,
                    enc_inits[i],
                    params_all[i],
                    ddm_sampler,
                    sc_aoas,
                    device,
                )
                gen_coords_list.append(gc)
                gen_aoas_list.append(torch.tensor(ga))
        all_gen_coords.append(torch.stack(gen_coords_list))  # [N, S, 2, 192]
        all_gen_aoas.append(torch.stack(gen_aoas_list))      # [N]
        print(f"  Pass {pass_i+1}/{args.n_passes} done.")

    gen_coords = torch.stack(all_gen_coords, dim=0).mean(0)  # [N, S, 2, 192]
    gen_aoas   = torch.stack(all_gen_aoas,   dim=0).mean(0)  # [N]

    # Normalise AoA from degrees to match scaler (scaler stored in degrees)
    # gt_aoas are already in degrees from the dataset
    # gen_aoas are already inverse-transformed inside generate_wing (output_decoded=True)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = compute_metrics(
        gen_coords, gt_coords,
        gen_aoas, gt_aoas,
        area_case_ratios=area_ratios,
    )

    print("\n=== Wing_TL Baseline Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Loss curve summary from checkpoint
    import torch as _torch
    ckpt = _torch.load(_DDM_CKPT, map_location="cpu", weights_only=False)
    stats = ckpt.get("stats", {})
    final_train_loss = float(stats["train_loss"][-1]) if len(stats.get("train_loss", [])) > 0 else None
    final_test_loss  = float(stats["test_loss"][-1])  if len(stats.get("test_loss",  [])) > 0 else None
    final_test_loss_x   = float(stats["test_loss_x"][-1])   if len(stats.get("test_loss_x",   [])) > 0 else None
    final_test_loss_aoa = float(stats["test_loss_AoA"][-1]) if len(stats.get("test_loss_AoA", [])) > 0 else None
    n_epochs = int(stats.get("epoch", 0)) + 1
    print(f"\n  Checkpoint training summary:")
    print(f"    epochs             : {n_epochs}")
    print(f"    final train loss   : {final_train_loss:.6f}")
    print(f"    final test loss    : {final_test_loss:.6f}")
    print(f"    final test loss_x  : {final_test_loss_x:.6f}")
    print(f"    final test loss_aoa: {final_test_loss_aoa:.6f}")

    # ── Save results ──────────────────────────────────────────────────────────
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"eval_wing_tl_baseline_{ts}")

    metrics["checkpoint"] = _DDM_CKPT
    metrics["n_passes"]   = args.n_passes
    metrics["n_test"]     = N
    metrics["n_slices"]   = S
    metrics["training"] = {
        "epochs": n_epochs,
        "final_train_loss": final_train_loss,
        "final_test_loss":  final_test_loss,
        "final_test_loss_x":   final_test_loss_x,
        "final_test_loss_aoa": final_test_loss_aoa,
    }

    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(base + ".txt", "w") as f:
        f.write(f"Wing_TL DDM Baseline Evaluation\n")
        f.write(f"Checkpoint : {_DDM_CKPT}\n")
        f.write(f"n_passes   : {args.n_passes}\n")
        f.write(f"n_test     : {N}\n")
        f.write(f"n_slices   : {S}\n\n")
        f.write(f"Training (from checkpoint stats):\n")
        f.write(f"  epochs             : {n_epochs}\n")
        f.write(f"  final train loss   : {final_train_loss:.6f}\n")
        f.write(f"  final test loss    : {final_test_loss:.6f}\n")
        f.write(f"  final test loss_x  : {final_test_loss_x:.6f}\n")
        f.write(f"  final test loss_aoa: {final_test_loss_aoa:.6f}\n\n")
        f.write(f"Metrics:\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.6f}\n")

    print(f"\nSaved to {base}.json / .txt")


if __name__ == "__main__":
    main()
