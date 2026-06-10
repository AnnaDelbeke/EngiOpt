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
        --wing_idx   0
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

def _draw_surface(ax, data: np.ndarray, title: str):
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
    ax.set_title(title, pad=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_zlim(-0.18, 0.18)
    ax.set_ylim(0, 1)
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)

    return surf


# ── Figure 1: exploded spanwise view ─────────────────────────────────────────

def plot_exploded_view(recon: np.ndarray, save_dir: str, scratch: str):
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
    ax.set_title("3D BAE — Exploded Spanwise Slice View", pad=10)
    ax.set_xlim(-0.05, 1.05);  ax.set_zlim(-0.18, 0.18);  ax.set_ylim(0, 1)
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)

    _save(fig, save_dir, scratch, "bae3d_exploded_view")
    plt.close()


# ── Figure 2: reconstruction surface ─────────────────────────────────────────

def plot_surface_mesh(recon: np.ndarray, save_dir: str, scratch: str):
    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection="3d")
    surf = _draw_surface(ax, recon, "3D BAE — Reconstructed Wing Surface")
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, aspect=20,
                 label="y/c  (surface height)")
    _save(fig, save_dir, scratch, "bae3d_surface_mesh")
    plt.close()


# ── Figure 3: GT vs reconstruction side-by-side ───────────────────────────────

def plot_gt_vs_recon(gt: np.ndarray, recon: np.ndarray,
                     save_dir: str, scratch: str):
    fig = plt.figure(figsize=(16, 5))
    ax_gt    = fig.add_subplot(121, projection="3d")
    ax_recon = fig.add_subplot(122, projection="3d")
    surf_gt  = _draw_surface(ax_gt,    gt,    "Ground Truth")
    _draw_surface(ax_recon, recon, "BAE Reconstruction")
    fig.colorbar(surf_gt, ax=[ax_gt, ax_recon], shrink=0.5, pad=0.04,
                 aspect=25, label="y/c  (surface height)")
    fig.suptitle("3D BAE — Ground Truth vs. Reconstruction",
                 fontsize=12, y=1.01)
    _save(fig, save_dir, scratch, "bae3d_gt_vs_recon")
    plt.close()


# ── GIFs: 360° rotating views ─────────────────────────────────────────────────

def plot_rotating_gifs(gt: np.ndarray, recon: np.ndarray,
                       save_dir: str, scratch: str,
                       n_frames: int = 72, fps: int = 20):
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    # GIF 1 — reconstruction only
    fig1 = plt.figure(figsize=(8, 5))
    ax1  = fig1.add_subplot(111, projection="3d")
    surf1 = _draw_surface(ax1, recon, "3D BAE — Reconstructed Wing Surface")
    fig1.colorbar(surf1, ax=ax1, shrink=0.45, pad=0.08, aspect=20,
                  label="y/c  (surface height)")

    def update1(frame):
        ax1.view_init(elev=22, azim=azimuths[frame])
        return []

    ani1 = FuncAnimation(fig1, update1, frames=n_frames,
                         interval=1000 // fps, blit=False)
    for base in (save_dir, scratch):
        path = os.path.join(base, "bae3d_surface_mesh_rotating.gif")
        ani1.save(path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved: {path}")
    plt.close(fig1)

    # GIF 2 — GT vs reconstruction
    fig2 = plt.figure(figsize=(16, 5))
    ax_gt    = fig2.add_subplot(121, projection="3d")
    ax_recon = fig2.add_subplot(122, projection="3d")
    surf_gt  = _draw_surface(ax_gt,    gt,    "Ground Truth")
    _draw_surface(ax_recon, recon, "BAE Reconstruction")
    fig2.colorbar(surf_gt, ax=[ax_gt, ax_recon], shrink=0.5, pad=0.04,
                  aspect=25, label="y/c  (surface height)")
    fig2.suptitle("3D BAE — Ground Truth vs. Reconstruction",
                  fontsize=12, y=1.01)

    def update2(frame):
        ax_gt.view_init(elev=22, azim=azimuths[frame])
        ax_recon.view_init(elev=22, azim=azimuths[frame])
        return []

    ani2 = FuncAnimation(fig2, update2, frames=n_frames,
                         interval=1000 // fps, blit=False)
    for base in (save_dir, scratch):
        path = os.path.join(base, "bae3d_gt_vs_recon_rotating.gif")
        ani2.save(path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved: {path}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir", default="results/bezier_ae_3d/run_039")
    p.add_argument("--wing_idx", type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(args.run_dir, "reconstructions_thesis")
    scratch  = "/cluster/scratch/adelbeke/thesis_figures"

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items    = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                    if item["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=0)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Validation wings: {len(val_dataset)}")

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
    print(f"Loaded run_039 (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    with torch.no_grad():
        x = val_dataset[args.wing_idx].unsqueeze(0).to(device)
        z = model.encode(x)
        y, _ = model.decode(z, return_cp=True)

    gt    = x.squeeze(0).cpu().numpy()
    recon = y.squeeze(0).cpu().numpy()
    print(f"Wing {args.wing_idx}: shape {recon.shape}")

    plot_exploded_view(recon, save_dir, scratch)
    plot_surface_mesh(recon, save_dir, scratch)
    plot_gt_vs_recon(gt, recon, save_dir, scratch)
    plot_rotating_gifs(gt, recon, save_dir, scratch)

    print(f"\nDone. All files in {save_dir}")


if __name__ == "__main__":
    main()
