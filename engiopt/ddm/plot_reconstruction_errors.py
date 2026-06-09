"""
Histogram of BAE→LVAE and BAE→PCA reconstruction errors over the test set.

For each wing in the test set:
  - Encode each slice through BAE → z [3, 30]
  - Encode all slices through LVAE → w [64] → decode back → z_rec [S, 3, 30]
    → BAE decode → coords_rec [S, 2, 192]
  - Encode flattened z through PCA → pca_z [64] → inverse → z_rec [S, 3, 30]
    → BAE decode → coords_rec [S, 2, 192]
  - Compute per-point L2 error against GT BAE reconstruction

Usage
-----
    python -m engiopt.ddm.plot_reconstruction_errors \
        --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
        --pca_checkpoint  results/ddm_pca/ddm_pca_v1_pca.pkl \
        [--seed 0]
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_w.train_ddm_w import Config, load_bae, load_lvae
from engiopt.ddm.ddm_pca.train_ddm_pca import Config as PCAConfig, load_bae as pca_load_bae
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def encode_wing_bae(coords, te_shifts, bae_model, device):
    """BAE-encode all slices of one wing. Returns z [S, 3, 30] and gt_coords [S, 2, 192]."""
    S = coords.shape[0]
    coords_c = coords.clone()
    coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
    te_x = coords_c[:, 0, 0]
    coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

    z_list, gt_list = [], []
    for s in range(S):
        x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
        z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
        dec, _, _ = bae_model.decode_z(z_s, z_ae_mode=True,
                                        denormalize_output=False, normalized_data=False)
        z_list.append(z_s.squeeze(0).cpu())
        gt_list.append(dec.squeeze(0).cpu())
    return torch.stack(z_list), torch.stack(gt_list)  # [S, 3, 30], [S, 2, 192]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lvae_checkpoint", type=str,
                   default="results/lvae/lae_dropout_0.25_flow_only_best.pth")
    p.add_argument("--pca_checkpoint",  type=str,
                   default="results/ddm_pca/ddm_pca_v1_pca.pkl")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results/plots")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config()
    cfg.lvae_checkpoint = args.lvae_checkpoint
    device = cfg.device

    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)
    lvae_params_scaler = getattr(lvae_model, 'scaler_params', None)

    with open(args.pca_checkpoint, 'rb') as f:
        pca = pickle.load(f)

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test     = list(new_dataset["test"])
    test_dataset = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    lvae_errors = []  # per-point L2 errors, all wings/slices/points
    pca_errors  = []

    bae_model.eval()
    lvae_model.encoder.eval()
    lvae_model.decoder.eval()

    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
            S = coords.shape[0]

            z_bae, gt_coords = encode_wing_bae(coords, te_shifts, bae_model, device)
            # z_bae: [S, 3, 30],  gt_coords: [S, 2, 192]

            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            if lvae_params_scaler is not None:
                flow_lvae_np = lvae_params_scaler.transform(flow_np)
            else:
                flow_lvae_np = flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)

            # --- LVAE round-trip ---
            z_wing = z_bae.unsqueeze(0).to(device)           # [1, S, 3, 30]
            w      = lvae_model.encoder(z_wing, flow_lvae)   # [1, w_dim]
            w_masked = lvae_model._apply_mask(w)
            z_rec_lvae, _, _, _, _ = lvae_model.decoder(w_masked, flow_lvae)  # [1, S, 3, 30]

            coords_lvae = []
            for s in range(S):
                dec, _, _ = bae_model.decode_z(
                    z_rec_lvae[0, s:s+1], z_ae_mode=True,
                    denormalize_output=False, normalized_data=False)
                coords_lvae.append(dec.squeeze(0).cpu())
            coords_lvae = torch.stack(coords_lvae)  # [S, 2, 192]

            err_lvae = ((coords_lvae - gt_coords) ** 2).sum(dim=1).sqrt()  # [S, 192]
            lvae_errors.append(err_lvae.numpy().ravel())

            # --- PCA round-trip ---
            z_flat = z_bae.flatten().unsqueeze(0).numpy()    # [1, S*3*30]
            z_pca  = pca.transform(z_flat)                   # [1, 64]
            z_rec_flat = pca.inverse_transform(z_pca)        # [1, S*3*30]
            z_rec_pca  = torch.tensor(z_rec_flat, dtype=torch.float32)
            z_rec_pca  = z_rec_pca.reshape(S, 3, 30).to(device)

            coords_pca = []
            for s in range(S):
                dec, _, _ = bae_model.decode_z(
                    z_rec_pca[s:s+1], z_ae_mode=True,
                    denormalize_output=False, normalized_data=False)
                coords_pca.append(dec.squeeze(0).cpu())
            coords_pca = torch.stack(coords_pca)  # [S, 2, 192]

            err_pca = ((coords_pca - gt_coords) ** 2).sum(dim=1).sqrt()  # [S, 192]
            pca_errors.append(err_pca.numpy().ravel())

    lvae_errors = np.concatenate(lvae_errors)
    pca_errors  = np.concatenate(pca_errors)

    print(f"LVAE recon L2  — mean: {lvae_errors.mean():.6f}  median: {np.median(lvae_errors):.6f}  p95: {np.percentile(lvae_errors, 95):.6f}")
    print(f"PCA  recon L2  — mean: {pca_errors.mean():.6f}  median: {np.median(pca_errors):.6f}  p95: {np.percentile(pca_errors, 95):.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, max(np.percentile(lvae_errors, 99), np.percentile(pca_errors, 99)), 80)

    for ax, errors, label, color in [
        (axes[0], lvae_errors, "LVAE (w-space)", "steelblue"),
        (axes[1], pca_errors,  "PCA (64-dim)",   "coral"),
    ]:
        ax.hist(errors, bins=bins, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(errors.mean(),           color='black',  lw=1.5, linestyle='--', label=f"mean={errors.mean():.4f}")
        ax.axvline(np.median(errors),       color='gray',   lw=1.5, linestyle=':',  label=f"median={np.median(errors):.4f}")
        ax.axvline(np.percentile(errors,95),color='red',    lw=1.0, linestyle='-',  label=f"p95={np.percentile(errors,95):.4f}")
        ax.set_title(f"BAE→{label} reconstruction L2 error (per point)", fontsize=10)
        ax.set_xlabel("L2 error (coordinate units)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4)

    fig.suptitle("Reconstruction error: LVAE vs PCA latent space (test set)", fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, "reconstruction_error_lvae_vs_pca.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
