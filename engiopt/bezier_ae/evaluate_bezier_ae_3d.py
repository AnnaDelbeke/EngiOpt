"""
Evaluate and plot reconstructions from BezierAutoencoder3D.

Plots a grid of all S spanwise slices (ground truth vs reconstruction)
for a few example wings, saved to results/bezier_ae_3d/<run>/reconstructions/.
Also prints and saves a per-span MSE table to reconstructions/spanwise_mse.csv.

Usage
-----
    # Default (uniform dataset, run_040):
    python -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
        --checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_040

    # Tip-biased dataset:
    python -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
        --checkpoint results/bezier_ae_3d/run_047/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_047 \
        --slices_pkl Wing_TL/data/processed/new_dataset_tip_biased_slices.pkl
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_DEFAULT_SLICES_PKL = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL        = "Wing_TL/data/processed/new_dataset_scalars.pkl"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_wing_reconstruction(x_wing: torch.Tensor, y_wing: torch.Tensor,
                             cp_wing: torch.Tensor,
                             example_idx: int, run_dir: str):
    """Plot all S slices: ground truth, reconstruction, and control polygon.

    x_wing / y_wing : [S, 2, 192]
    cp_wing         : [S, 2, n_cp]
    """
    S = x_wing.shape[0]
    ncols = 3
    nrows = (S + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten()

    for s in range(S):
        ax = axes[s]

        gt_x = x_wing[s, 0].numpy()
        gt_y = x_wing[s, 1].numpy()
        
        recon_x = y_wing[s, 0].numpy()
        recon_y = y_wing[s, 1].numpy()
        
        cp_x = cp_wing[s, 0].numpy()
        cp_y = cp_wing[s, 1].numpy()

        ax.plot(gt_x, gt_y,
                label="Ground Truth", linewidth=1.5, color="tab:blue")
        ax.plot(recon_x, recon_y,
                label="Reconstruction", linewidth=1.5, linestyle="--", color="tab:orange")
        ax.plot(cp_x, cp_y,
                marker="o", markersize=3, linestyle="--", linewidth=0.8,
                label="Control Polygon", color="tab:green", alpha=0.8)
        ax.set_title(f"Slice {s + 1}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        if s == 0:
            ax.legend(fontsize=7)

    for s in range(S, len(axes)):
        axes[s].set_visible(False)

    fig.suptitle(f"Wing {example_idx} — 3D BAE Reconstruction (all {S} spans)",
                 fontsize=12)
    plt.tight_layout()

    save_dir = os.path.join(run_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"wing_{example_idx:02d}_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


@torch.no_grad()
def plot_spanwise_error(x_wing: torch.Tensor, y_wing: torch.Tensor,
                        example_idx: int, run_dir: str):
    """Plot per-slice MSE across the span to show where error concentrates."""
    S = x_wing.shape[0]
    mse_per_slice = ((x_wing - y_wing) ** 2).mean(dim=(1, 2)).numpy()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, S + 1), mse_per_slice)
    ax.set_xlabel("Spanwise slice")
    ax.set_ylabel("MSE")
    ax.set_title(f"Wing {example_idx} — per-slice reconstruction MSE")
    ax.set_xticks(range(1, S + 1))

    save_dir = os.path.join(run_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"wing_{example_idx:02d}_spanwise_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir",    type=str,
                   default="results/bezier_ae_3d/run_040")
    p.add_argument("--slices_pkl", type=str, default=_DEFAULT_SLICES_PKL,
                   help="Path to slices pickle (use tip-biased pkl for tip-biased runs)")
    p.add_argument("--n_examples", type=int, default=5)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--cpx_bound", type=float, nargs=2, default=[0.0, 1.0])
    p.add_argument("--cpy_bound", type=float, nargs=2, default=[-0.75, 0.75])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Dataset (same split as training)
    print(f"Dataset: {args.slices_pkl}")
    new_dataset  = NewWingsDataset(args.slices_pkl, _SCALARS_PKL, seed=0)
    val_items    = [item for item in list(new_dataset["val"]) if item["final"] == 1]
    val_dataset  = WingsBezierDataset3D(val_items)
    print(f"Val wings: {len(val_dataset)}")

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_spans = val_dataset[0].shape[0]
    cpx_bound = ckpt.get("cpx_bound", args.cpx_bound)
    cpy_bound = ckpt.get("cpy_bound", args.cpy_bound)
    n_control_points  = ckpt.get("n_control_points", 32)
    slice_hidden_dims = ckpt.get("slice_hidden_dims", [256, 128])
    span_hidden_dims  = ckpt.get("span_hidden_dims",  [256, 128])
    latent_dim        = ckpt.get("latent_dim", args.latent_dim)
    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=n_control_points, n_data_points=192,
        slice_hidden_dims=slice_hidden_dims,
        span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim,
        cpx_bound=cpx_bound,
        cpy_bound=cpy_bound,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model (latent_dim={latent_dim}, n_spans={n_spans}, n_cp={n_control_points})")

    # Per-span MSE across all val wings
    per_span_mse  = np.zeros(n_spans)
    per_span_mse2 = np.zeros(n_spans)  # for std: E[x^2]
    total_mse = 0.0

    with torch.no_grad():
        for i in range(len(val_dataset)):
            x = val_dataset[i].unsqueeze(0).to(device)
            y, _ = model(x)
            sq = ((x - y) ** 2).squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # [S]
            per_span_mse  += sq
            per_span_mse2 += sq ** 2
            total_mse += sq.mean()

    per_span_mse  /= len(val_dataset)
    per_span_mse2 /= len(val_dataset)
    per_span_std   = np.sqrt(np.maximum(per_span_mse2 - per_span_mse ** 2, 0))
    overall_mse    = total_mse / len(val_dataset)

    print(f"\nOverall val MSE: {overall_mse:.6f}  (RMS: {overall_mse**0.5:.4f})")
    print(f"\n{'Span':>6}  {'MSE':>12}  {'RMS% chord':>11}  {'Std':>12}")
    for s in range(n_spans):
        print(f"  {s+1:4d}  {per_span_mse[s]:12.6f}  {per_span_mse[s]**0.5*100:11.4f}  {per_span_std[s]:12.6f}")

    # Save per-span MSE to CSV
    save_dir = os.path.join(args.run_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "spanwise_mse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["span", "mse", "rms_pct_chord", "std"])
        for s in range(n_spans):
            writer.writerow([s + 1, per_span_mse[s], per_span_mse[s]**0.5 * 100, per_span_std[s]])
    print(f"\nPer-span MSE saved to {csv_path}")

    # Per-example plots
    for i in range(min(args.n_examples, len(val_dataset))):
        x = val_dataset[i].unsqueeze(0).to(device)
        z   = model.encode(x)
        y, cp = model.decode(z, return_cp=True)

        x_wing  = x.squeeze(0).cpu()
        y_wing  = y.squeeze(0).cpu()
        cp_wing = cp.squeeze(0).cpu()

        wing_mse = ((x_wing - y_wing) ** 2).mean().item()
        print(f"Wing {i+1} MSE: {wing_mse:.6f}")

        plot_wing_reconstruction(x_wing, y_wing, cp_wing, i + 1, args.run_dir)
        plot_spanwise_error(x_wing, y_wing, i + 1, args.run_dir)

    print(f"\nAll plots saved to {args.run_dir}/reconstructions/")


if __name__ == "__main__":
    main()