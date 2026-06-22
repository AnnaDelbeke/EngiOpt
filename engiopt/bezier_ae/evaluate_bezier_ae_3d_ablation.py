"""Evaluate a BAE-3D ablation checkpoint on the held-out test set, save metrics JSON
and a midspan reconstruction plot for the first test wing."""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
MIDSPAN_IDX  = 7   # ~midspan for 15 slices (0-indexed)
WING_IDX     = 8   # classic airfoil shape, clear reconstruction error


def save_midspan_plot(x_wing, y_wing, n_samples, out_dir):
    """Save ground truth vs reconstruction for the midspan slice."""
    gt    = x_wing[MIDSPAN_IDX].numpy()   # [2, 192]
    recon = y_wing[MIDSPAN_IDX].numpy()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(gt[0],    gt[1],    color="#4477AA", lw=1.5, label="Ground truth")
    ax.plot(recon[0], recon[1], color="#EE6677", lw=1.5, linestyle="--", label="Reconstruction")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout(pad=0.2)
    path = os.path.join(out_dir, "midspan_recon.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_samples",  type=int, required=True)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out_dir",    type=str, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | n_samples={args.n_samples} seed={args.seed}")

    ds = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    test_items = [it for it in list(ds["test"]) if it["final"] == 1]
    test_ds = WingsBezierDataset3D(test_items)
    print(f"Test wings: {len(test_ds)}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_spans = test_ds[0].shape[0]
    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=32, n_data_points=192,
        slice_hidden_dims=ckpt.get("slice_hidden_dims", [256, 256, 256]),
        span_hidden_dims=ckpt.get("span_hidden_dims",  [256, 256]),
        latent_dim=ckpt.get("latent_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mses = []
    with torch.no_grad():
        for i, item in enumerate(test_ds):
            x = item.unsqueeze(0).to(device)
            y, _ = model(x)
            mse = ((x - y) ** 2).mean().item()
            mses.append(mse)
            if i == WING_IDX and args.seed == 0:
                save_midspan_plot(x.squeeze(0).cpu(), y.squeeze(0).cpu(),
                                  args.n_samples, args.out_dir)

    mean_mse = float(np.mean(mses))
    std_mse  = float(np.std(mses))
    print(f"Test MSE: {mean_mse:.6f} ± {std_mse:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "n_test": len(test_ds),
        "mse_mean": mean_mse,
        "mse_std": std_mse,
    }
    with open(f"{args.out_dir}/eval_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out_dir}/eval_metrics.json")


if __name__ == "__main__":
    main()
