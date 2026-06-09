"""
Plot sorted latent dimension std distribution for a given LVAE_3d checkpoint.
Shows the PCA-like importance ordering induced by the LV penalty.

Usage:
    python plot_latent_std_v16.py
    python plot_latent_std_v16.py --checkpoint results/lvae_3d/lvae_3d_v16.pth
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from engiopt.lvae.train_lvae_3d import LAEEncoderJoint3D
from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset


def compute_latent_stds(ckpt_path: str, bae_path: str, device: str = "cpu") -> np.ndarray:
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    bae_ckpt = torch.load(bae_path,  map_location=device, weights_only=False)

    encoder = LAEEncoderJoint3D(
        bae_latent_dim     = ckpt["bae_latent_dim"],
        c_dim              = 4,
        n_spans            = ckpt["n_spans"],
        pressure_length    = ckpt["pressure_length"],
        pressure_embed_dim = ckpt["pressure_embed_dim"],
        lae_latent_dim     = ckpt["lae_latent_dim"],
        dropout            = ckpt["dropout"],
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    bae = BezierAutoencoder3D(
        n_spans           = bae_ckpt["n_spans"],
        n_control_points  = 32,
        n_data_points     = 192,
        slice_hidden_dims = bae_ckpt["slice_hidden_dims"],
        span_hidden_dims  = bae_ckpt["span_hidden_dims"],
        latent_dim        = bae_ckpt["latent_dim"],
    )
    bae.load_state_dict(bae_ckpt["model_state_dict"])
    bae.eval()

    ds    = NewWingsDataset(
        "Wing_TL/data/processed/new_dataset_slices.pkl",
        "Wing_TL/data/processed/new_dataset_scalars.pkl",
        seed=0,
    )
    items = [x for x in list(ds["train"]) if x["final"] == 1]

    ws = []
    with torch.no_grad():
        for item in items:
            coords = torch.tensor(item["coords"],    dtype=torch.float32)
            te     = torch.tensor(item["te_shifts"], dtype=torch.float32)
            coords[:, :, 1] -= te.unsqueeze(1)
            te_x = coords[:, 0, 0]
            coords[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords[:, :, 0] = (coords[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            z_bae    = bae.encode(coords.permute(0, 2, 1).unsqueeze(0)).squeeze(0).unsqueeze(0)
            pressure = torch.tensor(item["coef_pressure"], dtype=torch.float32).unsqueeze(0)
            params   = torch.tensor(
                [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            ).unsqueeze(0)
            ws.append(encoder(z_bae, pressure, params).squeeze(0))

    ws   = torch.stack(ws)
    stds = ws.std(dim=0).numpy()
    return stds, ckpt["lae_latent_dim"]


def plot_stds(stds: np.ndarray, lae_latent_dim: int, out_path: str, title: str):
    stds_sorted = np.sort(stds)[::-1]
    mask_active = stds_sorted >= 0.02

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if a else "tomato" for a in mask_active]
    ax.bar(range(lae_latent_dim), stds_sorted, color=colors, width=0.8)
    ax.axhline(0.02, color="red",    linestyle="--", linewidth=1.0, label="prune threshold (0.02)")
    ax.axhline(0.05, color="orange", linestyle=":",  linewidth=1.0, label="0.05")
    ax.set_xlabel("Latent dimension (sorted by std, descending)")
    ax.set_ylabel("Standard deviation")
    ax.set_title(title)
    ax.legend()

    n_active = int(mask_active.sum())
    ratio    = stds_sorted[0] / stds_sorted[-1] if stds_sorted[-1] > 0 else float("inf")
    info = (
        f"Active (≥0.02): {n_active}/{lae_latent_dim}  |  "
        f"Max: {stds_sorted[0]:.4f}  Min: {stds_sorted[-1]:.4f}  "
        f"Ratio: {ratio:.1f}×"
    )
    ax.text(0.01, 0.97, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")
    print(info)
    print(f"Top 10 stds: {stds_sorted[:10].round(4)}")
    print(f"Bot 10 stds: {stds_sorted[-10:].round(4)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="results/lvae_3d/lvae_3d_v16.pth")
    p.add_argument("--bae_checkpoint", type=str,
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model_name = args.checkpoint.split("/")[-1].replace(".pth", "")
    out_path   = args.out or f"results/lvae_3d_evaluation/latent_std_{model_name}.png"

    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading {args.checkpoint} ...")
    stds, lae_latent_dim = compute_latent_stds(args.checkpoint, args.bae_checkpoint)
    plot_stds(stds, lae_latent_dim, out_path, title=f"Latent std ordering — {model_name}")


if __name__ == "__main__":
    main()
