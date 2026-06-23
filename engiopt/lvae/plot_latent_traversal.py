"""
Latent traversal for the LVAE3D (v29).

Sweeps three latent dimensions chosen from the LV-induced variance hierarchy:
  - top:    highest-variance active dimension (dominant global mode)
  - middle: median-variance active dimension
  - bottom: lowest-variance active dimension (fine-detail mode)

For each dimension a separate figure is produced (one file per dim).  The
columns are the N_STEPS traversal steps (−3σ … +3σ).  The rows are the three
flow regimes (subsonic / transonic / supersonic), each using its own reference
wing.

Each cell shows:
  • the airfoil profile at mid-span
  • the Cp distribution at mid-span (Cp axis inverted: negative = suction = top)

Output: {save_stem}_top1.pdf/png, {save_stem}_top2.pdf/png, {save_stem}_top3.pdf/png

Usage
-----
    python -m engiopt.lvae.plot_latent_traversal
    python -m engiopt.lvae.plot_latent_traversal \\
        --checkpoint results/lvae_3d/lvae_3d_v29_reproducing_best.pth \\
        --n_steps 7 \\
        --save_stem thesis/figures/latent_traversal
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import (
    encode_item_3d, load_bae_3d, load_lvae_3d,
)

plt.rcParams.update({
    "text.usetex":     False,
    "font.family":     "serif",
    "font.size":       13,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi":      150,
})

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"

N_STEPS    = 5
CMAP       = matplotlib.colors.LinearSegmentedColormap.from_list(
    "dark_div", ["#1a4f8a", "#6a3d9a", "#c0392b"]
)
REGIME_LABELS = ["Subsonic", "Transonic", "Supersonic"]
DIM_LABELS    = ["top", "middle", "bottom"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_dim_stds(model, z_baes, pressures, params, device, batch=64):
    """Per-latent-dim std across the dataset."""
    model.encoder.eval()
    all_w = []
    with torch.no_grad():
        for s in range(0, z_baes.shape[0], batch):
            e = min(s + batch, z_baes.shape[0])
            w = model.encoder(
                z_baes[s:e].to(device),
                pressures[s:e].to(device),
                params[s:e].to(device),
            )
            all_w.append(w.cpu())
    all_w = torch.cat(all_w, dim=0)
    return all_w.std(dim=0), all_w


def pick_traversal_dims(dim_stds, active_mask):
    """Return (top, median, bottom) active dim indices by variance."""
    active_indices = torch.where(active_mask)[0]
    active_stds    = dim_stds[active_mask]
    order          = torch.argsort(active_stds, descending=True)
    active_sorted  = active_indices[order]
    n = len(active_sorted)
    return [int(active_sorted[0]), int(active_sorted[n // 2]), int(active_sorted[-1])]


@torch.no_grad()
def decode_traversal(model, bae_model, w_ref, dim_idx, dim_std, params, device,
                     n_steps=7, n_sigma=3.0):
    """Sweep w[dim_idx] over [-n_sigma·σ, +n_sigma·σ], return shapes + Cp."""
    vals = torch.linspace(-n_sigma * dim_std, n_sigma * dim_std, n_steps)
    mask = model.active_latent_mask.to(device)
    coords_list, pressure_list = [], []

    for v in vals:
        w = w_ref.clone().to(device)
        w[0, dim_idx] = float(v)
        w_masked = w * mask.float()

        z_bae_pred, _aoa, _eta, pressure_pred_norm, _perf = model.decoder(w_masked, params)
        coords = bae_model.decode(z_bae_pred).cpu()      # [1, S, 2, 192]
        coords_list.append(coords[0])                     # [S, 2, 192]

        if pressure_pred_norm is not None and model.scaler_pressures is not None:
            p = model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
        elif pressure_pred_norm is not None:
            p = pressure_pred_norm.cpu()
        else:
            p = torch.zeros(coords[0].shape[0], 192)
        pressure_list.append(p[0] if p.ndim == 3 else p)  # [S, 192]

    return torch.stack(coords_list), torch.stack(pressure_list)



def _encode_ref(model, bae_model, item, device):
    """Encode one dataset item → (w_ref [1, D], params [1, 4])."""
    z_bae, _gt, _aoa, params_scaled, _te, gt_pressure, _le, _ch = encode_item_3d(
        item, bae_model, model, device, apply_x_norm=True,
    )
    params_d      = params_scaled.to(device)
    z_bae_d       = z_bae.unsqueeze(0).to(device)
    gt_pressure_d = gt_pressure.unsqueeze(0).to(device)
    with torch.no_grad():
        w_ref = model.encoder(z_bae_d, gt_pressure_d, params_d)
    return w_ref, params_d


def _find_regime_refs(test_dataset):
    """Return one item per regime: (subsonic, transonic, supersonic)."""
    sub = next((it for it in test_dataset if it["mach"] < 0.8),          None)
    tra = next((it for it in test_dataset if 0.8 <= it["mach"] < 1.0),   None)
    sup = next((it for it in test_dataset if it["mach"] >= 1.0),          None)
    return sub, tra, sup


# ---------------------------------------------------------------------------
# Per-dimension figure  (rows = regimes, columns = traversal steps)
# ---------------------------------------------------------------------------

def plot_one_dim(model, bae_model, ref_items, regime_labels, device,
                 dim_idx, dim_std, dim_label,
                 slice_idx=7, n_steps=7, n_sigma=3.0,
                 save_stem="thesis/figures/latent_traversal"):
    """One figure for one latent dimension, 3 regime rows × n_steps columns."""
    n_regimes = len(ref_items)

    # Layout constants (all in inches)
    col_w    = 1.55
    af_h_in  = 0.7    # airfoil panel height
    cp_h_in  = 1.4    # Cp panel height
    cell_h   = af_h_in + cp_h_in
    gap_in   = 0.10   # gap between airfoil and Cp within a cell
    row_sep  = 0.35   # vertical gap between regime rows
    cbar_w   = 0.18   # vertical colorbar width
    cbar_gap = 0.12   # gap between plot area and colorbar
    margin_l = 0.90   # left margin for regime label
    margin_r = cbar_gap + cbar_w + 0.25
    margin_t = 0.25   # top margin
    margin_b = 0.55   # bottom margin (x-labels)

    plot_area_w = n_steps * col_w
    fig_w = margin_l + plot_area_w + margin_r
    fig_h = margin_t + n_regimes * cell_h + (n_regimes - 1) * row_sep + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=-n_sigma, vmax=n_sigma))
    sm.set_array([])

    for row_i, (item, rlabel) in enumerate(zip(ref_items, regime_labels)):
        if item is None:
            continue

        w_ref, params_d = _encode_ref(model, bae_model, item, device)
        coords_t, pressure_t = decode_traversal(
            model, bae_model, w_ref, dim_idx, dim_std, params_d, device,
            n_steps=n_steps, n_sigma=n_sigma,
        )

        # Vertical origin of this row (from top of figure, in inches)
        row_top_in = margin_t + row_i * (cell_h + row_sep)

        # regime label to the left of the airfoil row
        fig.text(
            (margin_l - 0.05) / fig_w,
            1.0 - (row_top_in + af_h_in * 0.5) / fig_h,
            rlabel,
            ha="right", va="center", fontsize=11, fontweight="bold",
        )

        # ── per-step columns ──────────────────────────────────────────────────
        all_cp = pressure_t[:, slice_idx].numpy()   # [n_steps, 192]
        cp_min = float(all_cp.min())
        cp_max = float(all_cp.max())
        pad    = (cp_max - cp_min) * 0.08
        cp_ylim = (cp_min - pad, cp_max + pad)

        for col_i in range(n_steps):
            color = CMAP(col_i / (n_steps - 1))
            col_left_in = margin_l + col_i * col_w
            col_w_used  = col_w * 0.88

            # airfoil panel
            ax_af = fig.add_axes([
                col_left_in / fig_w,
                1.0 - (row_top_in + af_h_in) / fig_h,
                col_w_used / fig_w,
                af_h_in / fig_h,
            ])
            xy = coords_t[col_i, slice_idx].numpy()   # [2, 192]
            ax_af.plot(xy[0], xy[1], color=color, lw=0.9)
            ax_af.set_aspect("equal")
            ax_af.axis("off")

            # Cp panel
            ax_cp = fig.add_axes([
                col_left_in / fig_w,
                1.0 - (row_top_in + af_h_in + gap_in + cp_h_in) / fig_h,
                col_w_used / fig_w,
                cp_h_in / fig_h,
            ])

            cp    = all_cp[col_i]
            x_pts = xy[0]
            le_i  = int(np.argmin(x_pts))
            upper = np.zeros(len(x_pts), dtype=bool)
            upper[:le_i + 1] = True

            for surf_mask, ls in [(upper, "-"), (~upper, "--")]:
                if surf_mask.sum() == 0:
                    continue
                idx = np.argsort(x_pts[surf_mask])
                ax_cp.plot(x_pts[surf_mask][idx], cp[surf_mask][idx],
                           color=color, lw=0.9, linestyle=ls)

            ax_cp.set_xlim(-0.05, 1.05)
            ax_cp.set_ylim(*cp_ylim)
            ax_cp.tick_params(labelsize=9, length=2, pad=1)

            is_last_row = (row_i == n_regimes - 1)
            if is_last_row:
                ax_cp.set_xlabel("x/c", fontsize=10, labelpad=2)
            else:
                ax_cp.set_xticklabels([])

            if col_i == 0:
                ax_cp.set_ylabel("$C_p$", fontsize=10, labelpad=2)
            else:
                ax_cp.set_yticklabels([])

    # ── single vertical colorbar on the right ─────────────────────────────────
    fig.canvas.draw()
    # center the colorbar on the middle row, slightly taller than one cell
    mid_row_i      = n_regimes // 2
    mid_center_in  = margin_t + mid_row_i * (cell_h + row_sep) + cell_h * 0.5
    cbar_ht_in     = cell_h * 1.3
    cbar_left = (margin_l + plot_area_w + cbar_gap) / fig_w
    cbar_bot  = 1.0 - (mid_center_in + cbar_ht_in * 0.5) / fig_h
    cbar_ht   = cbar_ht_in / fig_h
    cbar_ax   = fig.add_axes([cbar_left, cbar_bot, cbar_w / fig_w, cbar_ht])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([-n_sigma, 0, n_sigma])
    cb.set_ticklabels([f"$-{n_sigma:.0f}\\sigma$", "$0$", f"$+{n_sigma:.0f}\\sigma$"])
    cb.ax.tick_params(labelsize=10)

    os.makedirs(os.path.dirname(save_stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        path = f"{save_stem}_{dim_label}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison figure: all top-k dims as rows, single regime, larger cells
# ---------------------------------------------------------------------------

def _plot_single_dim(coords_t, pressure_t, dim_idx, cp_ylim,
                     slice_idx, n_steps, n_sigma, save_path_stem):
    """One figure for one latent dimension: airfoil row + Cp row, n_steps columns."""
    vals_arr = np.linspace(-n_sigma, n_sigma, n_steps)
    all_cp   = pressure_t[:, slice_idx].numpy()

    fig, axes = plt.subplots(
        2, n_steps + 1,
        figsize=(n_steps * 2.0 + 1.2, 3.5),
        gridspec_kw={
            "height_ratios": [1.0, 2.5],
            "width_ratios":  [1.0] * n_steps + [0.08],
            "hspace": 0.0,
            "wspace": 0.18,
        },
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=-n_sigma, vmax=n_sigma))
    sm.set_array([])

    for col_i in range(n_steps):
        color = CMAP(col_i / (n_steps - 1))
        xy    = coords_t[col_i, slice_idx].numpy()

        # airfoil
        ax_af = axes[0, col_i]
        ax_af.plot(xy[0], xy[1], color=color, lw=1.8)
        ax_af.set_aspect("equal")
        ax_af.axis("off")

        # Cp
        ax_cp = axes[1, col_i]
        cp    = all_cp[col_i]
        x_pts = xy[0]
        le_i  = int(np.argmin(x_pts))
        upper = np.zeros(len(x_pts), dtype=bool)
        upper[:le_i + 1] = True
        for surf_mask, ls in [(upper, "-"), (~upper, "--")]:
            if surf_mask.sum() == 0:
                continue
            idx_s = np.argsort(x_pts[surf_mask])
            ax_cp.plot(x_pts[surf_mask][idx_s], cp[surf_mask][idx_s],
                       color=color, lw=1.0, linestyle=ls)
        ax_cp.set_xlim(-0.05, 1.05)
        ax_cp.set_xticks([0, 0.5, 1])
        ax_cp.set_xticklabels(["0", "0.5", "1"])
        ax_cp.set_ylim(*cp_ylim)
        ax_cp.tick_params(labelsize=11, length=3, pad=2)
        for sp in ["top", "right"]:
            ax_cp.spines[sp].set_visible(False)
        ax_cp.spines["left"].set_linewidth(0.6)
        ax_cp.spines["bottom"].set_linewidth(0.6)
        ax_cp.set_xlabel("x/c", fontsize=12, labelpad=2)
        if col_i == 0:
            ax_cp.set_ylabel("$C_p$", fontsize=12, labelpad=3)
        else:
            ax_cp.set_yticklabels([])
            ax_cp.spines["left"].set_visible(False)
            ax_cp.tick_params(left=False)

        axes[0, n_steps].axis("off")
        axes[1, n_steps].axis("off")

    # colorbar
    fig.canvas.draw()
    pos_top = axes[0, n_steps].get_position()
    pos_bot = axes[1, n_steps].get_position()
    cbar_ax = fig.add_axes([
        pos_top.x0 + 0.005,
        pos_bot.y0,
        pos_top.width * 1.8,
        pos_top.y1 - pos_bot.y0,
    ])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([-n_sigma, 0, n_sigma])
    cb.set_ticklabels([f"$-{n_sigma:.0f}\\sigma$", "$0$", f"$+{n_sigma:.0f}\\sigma$"])
    cb.ax.tick_params(labelsize=11)

    for ext in ("pdf", "png"):
        path = f"{save_path_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_comparison(model, bae_model, ref_item, regime_label, device,
                    traversal_dims,
                    slice_idx=7, n_steps=7, n_sigma=3.0,
                    save_stem="thesis/figures/latent_traversal"):
    """One figure per latent dimension (airfoil + Cp rows), shared Cp y-range."""
    # Precompute all traversals first so Cp y-range is shared across dims
    all_traversals = []
    for dim_idx, dim_std, dim_label in traversal_dims:
        w_ref, params_d = _encode_ref(model, bae_model, ref_item, device)
        coords_t, pressure_t = decode_traversal(
            model, bae_model, w_ref, dim_idx, dim_std, params_d, device,
            n_steps=n_steps, n_sigma=n_sigma,
        )
        all_traversals.append((coords_t, pressure_t, dim_idx, dim_std, dim_label))

    all_cp_vals = np.concatenate([
        pt[:, slice_idx].numpy().ravel() for _, pt, *_ in all_traversals
    ])
    cp_min  = float(all_cp_vals.min())
    cp_max  = float(all_cp_vals.max())
    pad     = (cp_max - cp_min) * 0.06
    cp_ylim = (cp_min - pad, cp_max + pad)

    os.makedirs(os.path.dirname(save_stem) or ".", exist_ok=True)
    dim_tags = ["top1", "top2", "top3"]
    for i, (coords_t, pressure_t, dim_idx, dim_std, dim_label) in enumerate(all_traversals):
        stem = f"{save_stem}_comparison_{dim_tags[i]}"
        _plot_single_dim(coords_t, pressure_t, dim_idx, cp_ylim,
                         slice_idx, n_steps, n_sigma, stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=LVAE_CKP)
    p.add_argument("--bae_checkpoint", default=BAE_CKP)
    p.add_argument("--bae_latent_dim", type=int, default=128)
    p.add_argument("--lae_latent_dim", type=int, default=64)
    p.add_argument("--n_spans",        type=int, default=15)
    p.add_argument("--slice_idx",      type=int, default=7,
                   help="Spanwise slice to display (0=root, n_spans-1=tip); default=7 (mid)")
    p.add_argument("--n_steps",        type=int, default=5)
    p.add_argument("--n_sigma",        type=float, default=3.0)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--save_stem",      default="thesis/figures/latent_traversal",
                   help="Files saved as {save_stem}_top/middle/bottom.pdf/png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bae_model = load_bae_3d(args.bae_checkpoint, device, latent_dim=args.bae_latent_dim)
    model     = load_lvae_3d(args.checkpoint, device, bae_model,
                             bae_latent_dim=args.bae_latent_dim,
                             lae_latent_dim=args.lae_latent_dim,
                             n_spans=args.n_spans)

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_dataset = [item for item in list(new_dataset["test"]) if item["final"] == 1]
    print(f"Test wings: {len(test_dataset)}")

    # ── regime reference wings ───────────────────────────────────────────────
    sub, tra, sup = _find_regime_refs(test_dataset)
    ref_items = [sub, tra, sup]
    for item, label in zip(ref_items, REGIME_LABELS):
        if item is not None:
            print(f"  {label}: Mach={item['mach']:.3f}, "
                  f"Re={item['reynolds']:.2e}, CL_target={item['cl_target']:.3f}")
        else:
            print(f"  {label}: no wing found in test set")

    # ── per-dim stds (test set) ──────────────────────────────────────────────
    print("Computing per-dim stds across test set...")
    z_baes_list, params_list, pressures_list = [], [], []
    for item in test_dataset:
        z_bae, _gt, _aoa, params_scaled, _te, pressure, _le, _ch = encode_item_3d(
            item, bae_model, model, device, apply_x_norm=True,
        )
        z_baes_list.append(z_bae)
        params_list.append(params_scaled.squeeze(0))
        pressures_list.append(pressure)

    z_baes_t    = torch.stack(z_baes_list)
    params_t    = torch.stack(params_list)
    pressures_t = torch.stack(pressures_list)

    dim_stds, _ = compute_dim_stds(model, z_baes_t, pressures_t, params_t, device)
    active_mask = model.active_latent_mask.cpu()
    n_active    = int(active_mask.sum().item())
    print(f"Active dims: {n_active}/{model.lae_latent_dim}")

    top_indices = pick_traversal_dims(dim_stds, active_mask)
    traversal_dims = [
        (d_idx, float(dim_stds[d_idx]), label)
        for d_idx, label in zip(top_indices, DIM_LABELS)
    ]
    for d_idx, d_std, d_label in traversal_dims:
        print(f"  {d_label}: dim {d_idx}  σ={d_std:.3f}")

    # ── produce one figure per latent dimension ──────────────────────────────
    for dim_idx, dim_std, dim_label in traversal_dims:
        print(f"\nPlotting traversal for {dim_label} dim ({dim_idx})...")
        plot_one_dim(
            model, bae_model,
            ref_items=ref_items,
            regime_labels=REGIME_LABELS,
            device=device,
            dim_idx=dim_idx,
            dim_std=dim_std,
            dim_label=dim_label,
            slice_idx=args.slice_idx,
            n_steps=args.n_steps,
            n_sigma=args.n_sigma,
            save_stem=args.save_stem,
        )

    # ── comparison figure: all 3 dims side-by-side (subsonic) ────────────────
    print("\nPlotting comparison figure (subsonic)...")
    plot_comparison(
        model, bae_model,
        ref_item=sub,
        regime_label=REGIME_LABELS[0],
        device=device,
        traversal_dims=traversal_dims,
        slice_idx=args.slice_idx,
        n_steps=args.n_steps,
        n_sigma=args.n_sigma,
        save_stem=args.save_stem,
    )


if __name__ == "__main__":
    main()
