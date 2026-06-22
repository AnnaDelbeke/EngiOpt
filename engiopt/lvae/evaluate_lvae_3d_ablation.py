"""
Evaluation script for the LVAE3D ablation study.

Loads a trained LVAE3D checkpoint, runs encode→decode on the test set,
computes the key metrics, and saves them to a JSON file compatible with
plot_ablation.py (same format as the DDM_W ablation eval).

Metrics saved:
  shape_mse     — BAE-roundtrip MSE in coordinate space (encode→decode→BAE-decode vs GT-BAE-roundtrip)
  pressure_mse  — pressure reconstruction MSE
  aoa_mse       — angle-of-attack reconstruction MSE
  mmd           — coordinate-space MMD (reconstructed vs GT distribution, per span averaged)
  mmd_w         — latent-space MMD (w-vectors vs N(0,I) prior)

Usage
-----
    python -m engiopt.lvae.evaluate_lvae_3d_ablation \
        --checkpoint     results/lvae_3d_ablation/n100_s0/lvae_3d_ablation_n100_s0_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --out_dir        results/lvae_3d_ablation/n100_s0
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.train_lvae_3d import LVAE3D
from engiopt.lvae.evaluate_lvae import (
    compute_geometry_metrics,
    compute_pressure_metrics,
    compute_latent_mmd,
    GAMMAS,
)
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def compute_mmd(a: torch.Tensor, b: torch.Tensor, gamma: float) -> float:
    n, m = a.shape[0], b.shape[0]
    Kaa = gaussian_kernel(a, a, gamma)
    Kbb = gaussian_kernel(b, b, gamma)
    Kab = gaussian_kernel(a, b, gamma)
    return (Kaa.sum() / (n * n) - 2 * Kab.sum() / (n * m) + Kbb.sum() / (m * m)).item()


def evaluate(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(args.checkpoint, device, bae_model)
    lvae_model.encoder.eval()
    lvae_model.decoder.eval()

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    test_dataset = [item for item in new_dataset["test"] if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    from engiopt.data_processing.utils import scaler
    scaler_params    = lvae_model.scaler_params
    scaler_aoas      = lvae_model.scaler_aoas
    scaler_pressures = lvae_model.scaler_pressures

    gt_coords_list    = []
    rec_coords_list   = []
    gt_pressures_list = []
    rec_pressures_list= []
    gt_aoas_list      = []
    rec_aoas_list     = []
    w_list            = []

    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],        dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"],     dtype=torch.float32)  # [S]
            pressure  = torch.tensor(item["coef_pressure"], dtype=torch.float32)  # [S, 192]
            aoa       = torch.tensor(float(item["alpha"]),  dtype=torch.float32)
            flow      = torch.tensor(
                [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            )

            # Normalise coordinates (same as training)
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

            # Scale params / pressure for encoder
            params_scaled = scaler_params.transform(flow.unsqueeze(0)).to(device) \
                if scaler_params else flow.unsqueeze(0).to(device)
            pressure_scaled = scaler_pressures.transform(pressure.unsqueeze(0)).to(device) \
                if scaler_pressures else pressure.unsqueeze(0).to(device)

            # Encode → w
            w = lvae_model.encoder(z_bae, pressure_scaled, params_scaled)  # [1, w_dim]

            # Decode → reconstructions
            z_bae_pred, aoa_pred, _, pressure_pred, _ = lvae_model.decoder(w, params_scaled)
            rec_recon = bae_model.decode(z_bae_pred).squeeze(0).cpu()      # [S, 2, 192]

            # Unscale pressure
            if pressure_pred is not None:
                pressure_rec = scaler_pressures.inverse_transform(pressure_pred).squeeze(0).cpu() \
                    if scaler_pressures else pressure_pred.squeeze(0).cpu()
            else:
                pressure_rec = torch.zeros_like(pressure)

            # Unscale aoa
            aoa_rec = scaler_aoas.inverse_transform(aoa_pred.cpu()) \
                if scaler_aoas else aoa_pred.cpu()

            gt_coords_list.append(gt_recon)
            rec_coords_list.append(rec_recon)
            gt_pressures_list.append(pressure)
            rec_pressures_list.append(pressure_rec)
            gt_aoas_list.append(aoa)
            rec_aoas_list.append(aoa_rec.squeeze())
            w_list.append(w.squeeze(0).cpu())

    gt_coords     = torch.stack(gt_coords_list)      # [N, S, 2, 192]
    rec_coords    = torch.stack(rec_coords_list)
    gt_pressures  = torch.stack(gt_pressures_list)   # [N, S, 192]
    rec_pressures = torch.stack(rec_pressures_list)
    gt_aoas       = torch.stack(gt_aoas_list)        # [N]
    rec_aoas      = torch.stack(rec_aoas_list)
    w_all         = torch.stack(w_list)              # [N, w_dim]

    N, S = gt_coords.shape[:2]

    # ── Shape MSE ────────────────────────────────────────────────────────────
    shape_mse = ((rec_coords - gt_coords) ** 2).mean().item()

    # ── Pressure MSE ─────────────────────────────────────────────────────────
    pressure_mse = ((rec_pressures - gt_pressures) ** 2).mean().item()

    # ── AoA MSE ──────────────────────────────────────────────────────────────
    aoa_mse = ((rec_aoas - gt_aoas) ** 2).mean().item()

    # ── Coordinate-space MMD (per span, averaged) ─────────────────────────────
    mmd_vals = []
    for s in range(S):
        gen_flat = rec_coords[:, s].reshape(N, -1)
        gt_flat  = gt_coords[:, s].reshape(N, -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
    mmd = float(np.mean(mmd_vals))

    # ── Latent-space MMD: w vs N(0,I) ────────────────────────────────────────
    # Rescale w to zero mean / unit std per dimension before applying the
    # fixed-bandwidth kernel, which otherwise underflows on w's scale.
    w_np = w_all.numpy()
    w_std = w_np.std(axis=0, keepdims=True)
    w_std[w_std < 1e-8] = 1.0
    w_scaled = (w_np - w_np.mean(axis=0, keepdims=True)) / w_std
    mmd_w = float(compute_latent_mmd(w_scaled))

    metrics = {
        "shape_mse":    shape_mse,
        "pressure_mse": pressure_mse,
        "aoa_mse":      aoa_mse,
        "mmd":          mmd,
        "mmd_w":        mmd_w,
        "checkpoint":   args.checkpoint,
        "n_test":       N,
    }

    print("\n=== LVAE Ablation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--bae_checkpoint", type=str,
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--out_dir",        type=str, default="results/lvae_3d_ablation")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = evaluate(args)

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.out_dir, f"eval_{stem}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
