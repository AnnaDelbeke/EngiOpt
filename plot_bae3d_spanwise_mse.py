"""
Aggregate spanwise MSE histogram for the 3D BAE.

For every wing in the val set, computes per-slice MSE [S], then averages
across all wings to show where along the span the model struggles most.

Usage
-----
    .venv/bin/python plot_bae3d_spanwise_mse.py
    .venv/bin/python plot_bae3d_spanwise_mse.py \
        --checkpoint results/bezier_ae_3d/run_013/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_013
"""

import argparse
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

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="results/bezier_ae_3d/run_041/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir", type=str,
                   default="results/bezier_ae_3d/run_041")
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--cpx_bound", type=float, nargs=2, default=[0.0, 1.0])
    p.add_argument("--cpy_bound", type=float, nargs=2, default=[-0.75, 0.75])
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load checkpoint first so n_spans is known before building the dataset
    ckpt              = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_spans           = ckpt.get("n_spans",           15)
    cpx_bound         = ckpt.get("cpx_bound",         args.cpx_bound)
    cpy_bound         = ckpt.get("cpy_bound",         args.cpy_bound)
    n_control_points  = ckpt.get("n_control_points",  32)
    slice_hidden_dims = ckpt.get("slice_hidden_dims",  [256, 128])
    span_hidden_dims  = ckpt.get("span_hidden_dims",   [256, 128])
    latent_dim        = ckpt.get("latent_dim",         args.latent_dim)

    # Build val set with matching number of spans (0 extra for 15-span, 3 for 18-span)
    num_extra_tip_slices = n_spans - 15
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items    = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                    if item["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=num_extra_tip_slices)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Val wings: {len(val_dataset)}")

    # Build model
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
    print(f"Model loaded (n_spans={n_spans}, latent_dim={latent_dim})")

    # Accumulate per-span MSE across all val wings
    span_mse_accum = np.zeros(n_spans)   # sum of per-span MSE
    span_mse_sq    = np.zeros(n_spans)   # for std dev
    n_wings = len(val_dataset)

    for i in range(n_wings):
        x = val_dataset[i].unsqueeze(0).to(device)   # [1, S, 2, 192]
        y, _ = model(x)
        # MSE per spanwise position: mean over (2 channels * 192 points)
        mse_per_span = ((x - y) ** 2).mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # [S]
        span_mse_accum += mse_per_span
        span_mse_sq    += mse_per_span ** 2
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_wings} wings...")

    mean_mse = span_mse_accum / n_wings
    std_mse  = np.sqrt(span_mse_sq / n_wings - mean_mse ** 2)

    span_indices = np.arange(1, n_spans + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, n_spans // 2), 4))
    bars = ax.bar(span_indices, mean_mse, yerr=std_mse, capsize=3,
                  color="steelblue", alpha=0.8, ecolor="gray")

    # Colour-code worst spans
    worst_k = max(1, n_spans // 5)
    worst_idx = np.argsort(mean_mse)[-worst_k:]
    for idx in worst_idx:
        bars[idx].set_color("tomato")

    ax.set_xlabel("Spanwise position (1 = root, last = tip)", fontsize=11)
    ax.set_ylabel("Mean reconstruction MSE", fontsize=11)
    ax.set_title(f"3D BAE — aggregate spanwise MSE over {n_wings} val wings\n"
                 f"(red = worst {worst_k} positions)", fontsize=12)
    ax.set_xticks(span_indices)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_dir = os.path.join(args.run_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aggregate_spanwise_mse.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")

    # Print summary table
    print(f"\n{'Span':>5}  {'Mean MSE':>10}  {'Std':>10}")
    for s in range(n_spans):
        print(f"{s+1:>5}  {mean_mse[s]:>10.6f}  {std_mse[s]:>10.6f}")


if __name__ == "__main__":
    main()
