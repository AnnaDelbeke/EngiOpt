"""
Thesis-quality reconstruction plot for the 2D (per-slice) BAE.

Shows three representative slices — root, mid-span, tip — each with
ground truth, reconstruction, and the Bézier control polygon.

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae2d
    python -m engiopt.bezier_ae.plot_thesis_bae2d \
        --checkpoint results/bezier_ae/run_006/models/bezier_ae_best.pt \
        --run_dir    results/bezier_ae/run_006
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engibench.problems.wings3D.v0 import Wings3D

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COL_GT    = "#2166ac"
COL_RECON = "#d6604d"
COL_CP    = "#4dac26"

N_SLICES  = 9   # Optiwing3D has 9 slices per wing


@torch.no_grad()
def plot_bae2d_three_slices(base_train_dataset, model, device: torch.device,
                             run_dir: str, wing_idx: int = 0, save_suffix: str = ""):
    """Plot root / mid-span / tip slices for a chosen final wing.

    wing_idx : index into the list of final wings in the train split.
    """
    # Collect all final items, each has coords [9, 192, 2]
    final_items = [item for item in base_train_dataset
                   if item.get("final", 0) == 1]
    print(f"Final wings available: {len(final_items)}")

    item = final_items[wing_idx]
    import numpy as np
    coords = np.array(item["coords"], dtype=np.float32)   # [9, 192, 2]

    # Apply same TE-centering as training
    import torch
    x = torch.tensor(coords).permute(0, 2, 1)             # [9, 2, 192]
    for s in range(x.shape[0]):
        x[s, 1, :] -= x[s, 1, 0]           # shift TE y to 0
        x[s, 0, :] += (1.0 - x[s, 0, 0])  # shift TE x to 1

    # root = slice 0, mid = slice N//2, tip = slice N-1
    indices = [0, N_SLICES // 2, N_SLICES - 1]
    labels  = ["Root", "Mid-span", "Tip"]

    # Pre-run all three slices to compute shared limits
    results = []
    for s_idx in indices:
        xi = x[s_idx].unsqueeze(0).to(device)   # [1, 2, 192]
        y, _, _, cp, _ = model(xi)
        results.append((
            xi[0, 0].cpu().numpy(), xi[0, 1].cpu().numpy(),
            y[0, 0].cpu().numpy(),  y[0, 1].cpu().numpy(),
            cp[0, 0].cpu().numpy(), cp[0, 1].cpu().numpy(),
        ))

    import numpy as np
    # Base y-limits on GT + reconstruction only — control polygon can extend
    # well beyond the airfoil and would squash the view if included
    all_y_airfoil = np.concatenate([r[1] for r in results] +
                                    [r[3] for r in results])
    y_margin = 0.04
    y_lo = all_y_airfoil.min() - y_margin
    y_hi = all_y_airfoil.max() + y_margin
    x_lo, x_hi = -0.02, 1.02

    fig, axes = plt.subplots(1, 3, figsize=(13, 2.8),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    for col, ((gt_x, gt_y, recon_x, recon_y, cp_x, cp_y), label) in \
            enumerate(zip(results, labels)):
        ax = axes[col]

        ax.plot(gt_x,    gt_y,    color=COL_GT,    lw=2.0,
                label="Ground truth", zorder=3)
        ax.plot(recon_x, recon_y, color=COL_RECON, lw=2.0, ls="--",
                label="Reconstruction", zorder=3)
        ax.plot(cp_x,    cp_y,    color=COL_CP,    lw=1.0, ls="--",
                marker="o", markersize=5, alpha=0.85,
                label="Control polygon", zorder=2)

        ax.set_title(label, pad=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if col == 0:
            ax.legend(loc="upper right", framealpha=0.9,
                      handlelength=1.6, borderpad=0.5)

    fig.suptitle("2D BAE — airfoil reconstruction at three spanwise positions",
                 fontsize=11)

    save_dir = os.path.join(run_dir, "reconstructions_thesis")
    os.makedirs(save_dir, exist_ok=True)
    stem = f"bae2d_wing{wing_idx}_thesis" if not save_suffix else save_suffix
    path = os.path.join(save_dir, f"{stem}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="results/bezier_ae/run_006/models/bezier_ae_best.pt")
    p.add_argument("--run_dir", type=str,
                   default="results/bezier_ae/run_006")
    p.add_argument("--wing_idx", type=int, default=0,
                   help="Index into the list of final wings in the train split")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset ───────────────────────────────────────────────────────────────
    problem           = Wings3D(seed=0)
    base_train_dataset = problem.dataset["train"]

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BezierAutoencoder(
        n_control_points=32,
        n_data_points=192,
        auto_batch=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded 2D BAE checkpoint.")

    plot_bae2d_three_slices(base_train_dataset, model, device,
                             args.run_dir, wing_idx=args.wing_idx)
    print(f"\nPlot saved to {args.run_dir}/reconstructions_thesis/")


if __name__ == "__main__":
    main()
