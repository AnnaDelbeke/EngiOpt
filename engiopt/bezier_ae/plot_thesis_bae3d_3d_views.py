"""
Thesis figures for the 3D joint BAE — 3D perspective plots + rotating GIFs.

Figure 1: Exploded spanwise view — 15 airfoil slices offset along the span.
Figure 2: Reconstructed 3D surface mesh.
Figure 3: Side-by-side GT vs reconstruction surface mesh.
GIF 1:    360° rotating reconstruction surface.
GIF 2:    360° rotating GT vs reconstruction side-by-side.

Usage
-----
    python -m engiopt.bezier_ae.plot_thesis_bae3d_3d_views
    python -m engiopt.bezier_ae.plot_thesis_bae3d_3d_views \
        --checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --run_dir    results/bezier_ae_3d/run_039 \
        --wing_indices 0 1 2 3 4
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from torch.utils.data import random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

# True span-to-chord ratio (span tip ~= 2.505 m, chord normalised to 1) so the
# 3D wing renders are not visually squashed into a cube.
_SPAN_CHORD_RATIO = 2.505

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":     150,
})


def _span_colours(n, cmap_name="coolwarm"):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def _save(fig, save_dir, scratch, stem, exts=("pdf", "png")):
    for ext in exts:
        for base in (save_dir, scratch):
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"Saved: {path}")


# ── shared surface drawing helper ─────────────────────────────────────────────

def _draw_surface(ax, data: np.ndarray, title: str | None = None):
    """Draw a surface mesh on ax. Returns the surface artist."""
    S, _, N = data.shape
    span_pos = np.linspace(0.0, 1.0, S)

    X = data[:, 0, :]
    Y = np.tile(span_pos[:, None], (1, N))
    Z = data[:, 1, :]

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="coolwarm",
        alpha=0.72,
        linewidth=0,
        antialiased=True,
        rcount=S,
        ccount=64,
        vmin=Z.min(),
        vmax=Z.max(),
    )

    colours = _span_colours(S)
    for s in range(S):
        ax.plot(data[s, 0], np.full(N, span_pos[s]), data[s, 1],
                color=colours[s], lw=0.8, alpha=0.6)

    ax.set_xlabel("x/c", labelpad=6)
    ax.set_ylabel("Span η", labelpad=6)
    ax.set_zlabel("y/c", labelpad=6)
    if title:
        ax.set_title(title, pad=10)
    # Use equal data ranges on all three axes so the wing is not distorted.
    # Chord x/c spans 1.1; pad span and thickness to the same range.
    ax.set_xlim(-0.05, 1.05)          # range = 1.1
    ax.set_ylim(-0.05, 1.05)          # range = 1.1 (span, centred on [0,1])
    ax.set_zlim(-0.55, 0.55)          # range = 1.1 (thickness, centred on 0)
    ax.set_box_aspect([1, _SPAN_CHORD_RATIO, 1])
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)

    return surf


# ── Figure 1: exploded spanwise view ─────────────────────────────────────────

def plot_exploded_view(recon: np.ndarray, save_dir: str, scratch: str, stem: str = "bae3d_exploded_view"):
    S = recon.shape[0]
    colours  = _span_colours(S)
    span_pos = np.linspace(0.0, 1.0, S)

    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection="3d")

    for s in range(S):
        x = recon[s, 0]
        y = recon[s, 1]
        z = np.full_like(x, span_pos[s])
        ax.plot(x, z, y, color=colours[s], lw=1.4, alpha=0.9)

    # TE and LE connector lines
    te_x  = recon[:, 0, 0];  te_y  = recon[:, 1, 0]
    le_idx = np.argmin(recon[:, 0, :], axis=1)
    le_x  = recon[np.arange(S), 0, le_idx]
    le_y  = recon[np.arange(S), 1, le_idx]
    ax.plot(te_x, span_pos, te_y, color="#555555", lw=0.7, ls="--", alpha=0.5)
    ax.plot(le_x, span_pos, le_y, color="#555555", lw=0.7, ls="--", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08, aspect=20)
    cbar.set_label("Normalised span  (root → tip)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Root", "Mid", "Tip"])

    ax.set_xlabel("x/c", labelpad=6)
    ax.set_ylabel("Span η", labelpad=6)
    ax.set_zlabel("y/c", labelpad=6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect([1, _SPAN_CHORD_RATIO, 1])
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)

    _save(fig, save_dir, scratch, stem)
    plt.close()


# ── Figure 2: reconstruction surface ─────────────────────────────────────────

def plot_surface_mesh(recon: np.ndarray, save_dir: str, scratch: str, stem: str = "bae3d_surface_mesh"):
    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection="3d")
    surf = _draw_surface(ax, recon)
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, aspect=20,
                 label="y/c  (surface height)")
    _save(fig, save_dir, scratch, stem)
    plt.close()


# ── Figure 3: GT vs reconstruction side-by-side ───────────────────────────────

def plot_gt_vs_recon(gt: np.ndarray, recon: np.ndarray,
                     save_dir: str, scratch: str, stem: str = "bae3d_gt_vs_recon"):
    """Saves GT and reconstruction as two separate untitled images, so the
    thesis can caption them as (a)/(b) subfigures in LaTeX instead of baking
    titles into the matplotlib figure."""
    for suffix, data in (("gt", gt), ("recon", recon)):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = _draw_surface(ax, data)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, aspect=20,
                     label="y/c  (surface height)")
        _save(fig, save_dir, scratch, f"{stem}_{suffix}")
        plt.close(fig)


# ── GIFs: 360° rotating views ─────────────────────────────────────────────────

def plot_rotating_gifs(gt: np.ndarray, recon: np.ndarray,
                       save_dir: str, scratch: str,
                       stem_mesh: str = "bae3d_surface_mesh_rotating",
                       stem_gtrecon: str = "bae3d_gt_vs_recon_rotating",
                       n_frames: int = 36, fps: int = 12):
    """36 frames at 12fps = 3s loop, much lower memory than 72@20."""
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    # GIF 1 — reconstruction only
    fig1 = plt.figure(figsize=(6, 4))
    ax1  = fig1.add_subplot(111, projection="3d")
    surf1 = _draw_surface(ax1, recon)
    fig1.colorbar(surf1, ax=ax1, shrink=0.45, pad=0.08, aspect=20,
                  label="y/c  (surface height)")

    def update1(frame):
        ax1.view_init(elev=22, azim=azimuths[frame])
        return []

    ani1 = FuncAnimation(fig1, update1, frames=n_frames,
                         interval=1000 // fps, blit=False)
    for base in (save_dir, scratch):
        path = os.path.join(base, f"{stem_mesh}.gif")
        ani1.save(path, writer="pillow", fps=fps, dpi=80)
        print(f"Saved: {path}")
    plt.close(fig1)

    # GIF 2 — GT vs reconstruction (smaller to keep memory down)
    fig2 = plt.figure(figsize=(10, 4))
    ax_gt    = fig2.add_subplot(121, projection="3d")
    ax_recon = fig2.add_subplot(122, projection="3d")
    surf_gt  = _draw_surface(ax_gt,    gt,    "Ground Truth")
    _draw_surface(ax_recon, recon, "BAE Reconstruction")
    fig2.colorbar(surf_gt, ax=[ax_gt, ax_recon], shrink=0.5, pad=0.04,
                  aspect=25, label="y/c  (surface height)")
    fig2.suptitle("3D BAE — Ground Truth vs. Reconstruction",
                  fontsize=11, y=1.01)

    def update2(frame):
        ax_gt.view_init(elev=22, azim=azimuths[frame])
        ax_recon.view_init(elev=22, azim=azimuths[frame])
        return []

    ani2 = FuncAnimation(fig2, update2, frames=n_frames,
                         interval=1000 // fps, blit=False)
    for base in (save_dir, scratch):
        path = os.path.join(base, f"{stem_gtrecon}.gif")
        ani2.save(path, writer="pillow", fps=fps, dpi=80)
        print(f"Saved: {path}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir", default="results/bezier_ae_3d/run_039")
    p.add_argument("--wing_indices", type=int, nargs="+", default=None)
    p.add_argument("--case_nums", type=int, nargs="+", default=None,
                   help="Plot these case numbers (looked up in the val set). "
                        "If both --wing_indices and --case_nums are given, they are combined.")
    p.add_argument("--slices_pkl",  default=_SLICES_PKL)
    p.add_argument("--scalars_pkl", default=_SCALARS_PKL)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(args.run_dir, "reconstructions_thesis")
    scratch  = "/cluster/scratch/adelbeke/thesis_figures"

    new_dataset  = NewWingsDataset(args.slices_pkl, args.scalars_pkl, seed=0)
    all_items    = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                    if item["final"] == 1]
    all_case_nums = [item["case_num"] for item in all_items]
    full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=0)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    generator    = torch.Generator().manual_seed(0)
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )
    val_case_nums = [all_case_nums[i] for i in val_dataset.indices]
    print(f"Validation wings: {len(val_dataset)}")

    # Resolve indices from --case_nums
    indices = list(args.wing_indices) if args.wing_indices else []
    if args.case_nums:
        for cn in args.case_nums:
            if cn in val_case_nums:
                indices.append(val_case_nums.index(cn))
            else:
                print(f"WARNING: case {cn} not found in val set — skipping")
    if not indices:
        indices = [0, 1, 2, 3, 4]
    # Deduplicate while preserving order
    seen = set()
    indices = [i for i in indices if not (i in seen or seen.add(i))]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BezierAutoencoder3D(
        n_spans=ckpt["n_spans"],
        n_control_points=ckpt["n_control_points"],
        n_data_points=192,
        slice_hidden_dims=ckpt["slice_hidden_dims"],
        span_hidden_dims=ckpt["span_hidden_dims"],
        latent_dim=ckpt["latent_dim"],
        cpx_bound=ckpt["cpx_bound"],
        cpy_bound=ckpt["cpy_bound"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    for wing_idx in indices:
        case_num = val_case_nums[wing_idx]
        print(f"\n── Wing {wing_idx}  (case {case_num}) ──")
        with torch.no_grad():
            x = val_dataset[wing_idx].unsqueeze(0).to(device)
            z = model.encode(x)
            y, _ = model.decode(z, return_cp=True)

        gt    = x.squeeze(0).cpu().numpy()
        recon = y.squeeze(0).cpu().numpy()
        print(f"  shape {recon.shape}")

        sfx = f"_case{case_num:05d}"
        plot_exploded_view(recon, save_dir, scratch, stem=f"bae3d_exploded_view{sfx}")
        plot_surface_mesh(recon, save_dir, scratch, stem=f"bae3d_surface_mesh{sfx}")
        plot_gt_vs_recon(gt, recon, save_dir, scratch, stem=f"bae3d_gt_vs_recon{sfx}")
        plot_rotating_gifs(
            gt, recon, save_dir, scratch,
            stem_mesh=f"bae3d_surface_mesh_rotating{sfx}",
            stem_gtrecon=f"bae3d_gt_vs_recon_rotating{sfx}",
        )

    print(f"\nDone. All files in {save_dir}")


if __name__ == "__main__":
    main()
