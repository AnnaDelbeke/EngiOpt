"""
PCA-only ablation: fit PCA on N training wings (BAE latents), evaluate
reconstruction quality on the test set, save metrics as JSON.

Metrics:
  shape_mse  — coordinate-space MSE (PCA roundtrip vs BAE-roundtrip GT)
  mmd        — coordinate-space MMD (reconstructed vs GT, per span averaged)
  explained_variance — cumulative explained variance ratio of the fitted PCA

Usage
-----
    python -m engiopt.ddm.ddm_pca.evaluate_pca_ablation_3d \
        --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --n_samples      100 \
        --n_components   64 \
        --seed           0 \
        --out_dir        results/pca_ablation/n100_s0
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.decomposition import PCA

from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

GAMMAS = [0.5, 25, 50, 100]


def gaussian_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def compute_mmd(a, b, gamma):
    n, m = a.shape[0], b.shape[0]
    Kaa = gaussian_kernel(a, a, gamma)
    Kbb = gaussian_kernel(b, b, gamma)
    Kab = gaussian_kernel(a, b, gamma)
    return (Kaa.sum() / (n * n) - 2 * Kab.sum() / (n * m) + Kbb.sum() / (m * m)).item()


def encode_wings(dataset_items, bae_model, device):
    """BAE-encode a list of dataset items → [N, bae_latent_dim] and GT coords."""
    z_list, coords_list = [], []
    bae_model.eval()
    with torch.no_grad():
        for item in dataset_items:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x  = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            x_wing   = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)  # [1, S, 2, 192]
            z_bae    = bae_model.encode(x_wing)                            # [1, bae_latent_dim]
            gt_recon = bae_model.decode(z_bae).squeeze(0).cpu()            # [S, 2, 192]

            z_list.append(z_bae.squeeze(0).cpu())
            coords_list.append(gt_recon)

    return torch.stack(z_list), torch.stack(coords_list)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bae_checkpoint", type=str, required=True)
    p.add_argument("--n_samples",      type=int, default=0,
                   help="Training wings to fit PCA on (0 = all)")
    p.add_argument("--n_components",   type=int, default=64)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--out_dir",        type=str, default="results/pca_ablation")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  |  n_samples={args.n_samples}  seed={args.seed}")

    bae_model = load_bae_3d(args.bae_checkpoint, device)

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_train    = list(new_dataset["train"])
    base_dataset = [item for item in all_train if item["final"] == 1]
    test_dataset = [item for item in new_dataset["test"] if item["final"] == 1]

    if args.n_samples > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(base_dataset), size=min(args.n_samples, len(base_dataset)), replace=False)
        base_dataset = [base_dataset[i] for i in sorted(idx)]

    print(f"Train: {len(base_dataset)}  |  Test: {len(test_dataset)}")

    # Encode training set
    print("Encoding training wings...")
    z_train, _ = encode_wings(base_dataset, bae_model, device)

    # Fit PCA
    print(f"Fitting PCA ({args.n_components} components)...")
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    pca.fit(z_train.numpy())
    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"Explained variance: {explained_var * 100:.1f}%")

    pca_path = os.path.join(args.out_dir, f"pca_n{args.n_samples}_s{args.seed}.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    # Encode test set
    print("Encoding test wings...")
    z_test, gt_coords = encode_wings(test_dataset, bae_model, device)  # [N, bae_latent_dim], [N, S, 2, 192]

    # PCA roundtrip
    z_pca      = pca.transform(z_test.numpy())          # [N, n_components]
    z_rec_np   = pca.inverse_transform(z_pca)           # [N, bae_latent_dim]
    z_rec      = torch.tensor(z_rec_np, dtype=torch.float32)

    print("Decoding reconstructed latents...")
    rec_coords_list = []
    bae_model.eval()
    with torch.no_grad():
        for i in range(z_rec.shape[0]):
            z = z_rec[i].unsqueeze(0).to(device)
            rec = bae_model.decode(z).squeeze(0).cpu()  # [S, 2, 192]
            rec_coords_list.append(rec)
    rec_coords = torch.stack(rec_coords_list)            # [N, S, 2, 192]

    # Metrics
    shape_mse = ((rec_coords - gt_coords) ** 2).mean().item()

    N, S = gt_coords.shape[:2]
    mmd_vals = []
    for s in range(S):
        gen_flat = rec_coords[:, s].reshape(N, -1)
        gt_flat  = gt_coords[:, s].reshape(N, -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
    mmd = float(np.mean(mmd_vals))

    metrics = {
        "shape_mse":        shape_mse,
        "mmd":              mmd,
        "explained_variance": explained_var,
        "n_components":     args.n_components,
        "n_train":          len(base_dataset),
        "n_test":           N,
    }

    print("\n=== PCA Ablation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"eval_pca_n{args.n_samples}_s{args.seed}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
