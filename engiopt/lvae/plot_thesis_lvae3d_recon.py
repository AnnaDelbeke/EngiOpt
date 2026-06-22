"""
Thesis-quality shape + Cp reconstruction plot for the joint LVAE (v29).

Mirrors the layout of plot_airfoil_cp_comparison() in plotting.py (used for
the raw evaluate_lvae_3d.py dumps), but restricted to a handful of
representative wings and saved as a clean PDF for the thesis.

Each wing is one flow regime and is saved as its own file ('{save_stem}_a.pdf',
'_b.pdf', '_c.pdf', ...) so the regimes can be reordered or repositioned
independently in the thesis. No per-axes header; a single shared legend
(Ground truth vs. Reconstructed) is placed below each figure's panels.

Usage
-----
    # Main results figure: one file per wing/regime, 6 slices, 2 rows of 3
    python -m engiopt.lvae.plot_thesis_lvae3d_recon

    # Appendix: one file per wing, all 15 slices wrapped into 2 rows of 8/7
    python -m engiopt.lvae.plot_thesis_lvae3d_recon \
        --full_slices --slices_per_row 8 \
        --save_stem thesis/figures/lvae_recon_shape_cp_appendix
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import (
    encode_item_3d, load_bae_3d, load_lvae_3d, reconstruct_batch_3d,
)

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
})

COL_GT    = "steelblue"
COL_RECON = "coral"


_WING_LABELS = "abcdefghijklmnopqrstuvwxyz"


def plot_recon(gt_airfoils, rec_airfoils, gt_pressures, rec_pressures,
               wing_indices, slice_indices, save_stem, pred_label="Reconstructed",
               slices_per_row=None, n_cols_per_row=None, eta_values=None):
    """Save one PDF/PNG per wing."""
    per_row = slices_per_row or n_cols_per_row or len(slice_indices)
    for wi_idx, wi in enumerate(wing_indices):
        chunks = [slice_indices[i:i + per_row]
                  for i in range(0, len(slice_indices), per_row)]
        _plot_recon_rows(
            gt_airfoils, rec_airfoils, gt_pressures, rec_pressures,
            wi=wi, wing_label=_WING_LABELS[wi_idx], slice_chunks=chunks,
            save_stem=f"{save_stem}_{_WING_LABELS[wi_idx]}", pred_label=pred_label,
            eta_values=eta_values,
        )


def _plot_recon_rows(gt_airfoils, rec_airfoils, gt_pressures, rec_pressures,
                     wi, wing_label, slice_chunks, save_stem, pred_label="Reconstructed",
                     eta_values=None):
    """Stack one wing's slices into multiple shape+Cp row-pairs (one per chunk)."""
    import matplotlib.gridspec as mgridspec

    n_cols = max(len(chunk) for chunk in slice_chunks)
    n_chunks = len(slice_chunks)

    fig = plt.figure(figsize=(2.2 * n_cols, 1.8 * n_chunks), constrained_layout=False)
    # Outer grid: one row per chunk, generous spacing between chunks
    outer_gs = mgridspec.GridSpec(n_chunks, 1, figure=fig, hspace=0.55,
                                  left=0.1, right=0.98, top=0.97, bottom=0.15)

    all_axes = []
    for chunk_idx in range(n_chunks):
        # Inner grid: airfoil row + Cp row, tight spacing within chunk
        inner_gs = mgridspec.GridSpecFromSubplotSpec(
            2, n_cols,
            subplot_spec=outer_gs[chunk_idx],
            height_ratios=[0.6, 1.0],
            hspace=0.08,
            wspace=0.32,
        )
        row_shape = []
        row_cp = []
        for col in range(n_cols):
            row_shape.append(fig.add_subplot(inner_gs[0, col]))
            row_cp.append(fig.add_subplot(inner_gs[1, col]))
        all_axes.append((row_shape, row_cp))

    legend_handles = None
    for chunk_idx, chunk in enumerate(slice_chunks):
        ax_shape_row, ax_cp_row = all_axes[chunk_idx]

        for col, sl in enumerate(chunk):
            eta = eta_values[sl] if eta_values is not None else None
            h = _draw_panel_pair(
                ax_shape_row[col], ax_cp_row[col],
                gt_airfoils, rec_airfoils, gt_pressures, rec_pressures,
                wi=wi, sl=sl, pred_label=pred_label, col=col, eta=eta,
            )
            if legend_handles is None:
                legend_handles = h
        for col in range(len(chunk), n_cols):
            ax_shape_row[col].axis("off")
            ax_cp_row[col].axis("off")

    # shape handles (solid lines) + upper/lower Cp handles
    h_gt, h_rec, h_gt_up, h_gt_lo, h_rec_up, h_rec_lo = legend_handles
    leg_handles = [h_gt, h_rec, h_gt_up, h_gt_lo, h_rec_up, h_rec_lo]
    leg_labels  = [
        "Ground truth (shape)", f"{pred_label} (shape)",
        "Ground truth (upper)", "Ground truth (lower)",
        f"{pred_label} (upper)", f"{pred_label} (lower)",
    ]
    fig.legend(leg_handles, leg_labels,
              loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
              fontsize=9, frameon=False)
    _savefig(fig, save_stem)


def _draw_panel_pair(ax_shape, ax_cp, gt_airfoils, rec_airfoils, gt_pressures,
                     rec_pressures, wi, sl, pred_label, col, eta=None):
    """Draw one shape+Cp panel pair. Returns legend handles."""
    # ── airfoil shape ────────────────────────────────────────────────────
    gt_xy  = gt_airfoils[wi, sl].numpy()    # [2, 192]
    rec_xy = rec_airfoils[wi, sl].numpy()

    h_gt,  = ax_shape.plot(gt_xy[0],  gt_xy[1],  color=COL_GT,    lw=1.2, label="Ground truth")
    h_rec, = ax_shape.plot(rec_xy[0], rec_xy[1], color=COL_RECON, lw=1.2, ls="--",
                           label=pred_label, zorder=3)

    ax_shape.set_aspect("equal")
    ax_shape.set_xticks([]); ax_shape.set_yticks([])
    if col == 0:
        ax_shape.set_ylabel("airfoil", fontsize=8)
    if eta is not None:
        ax_shape.set_title(f"$\\eta = {eta:.2f}$", fontsize=7, pad=5)

    # ── Cp distribution ─────────────────────────────────────────────────
    gt_cp  = gt_pressures[wi, sl].numpy()
    rec_cp = rec_pressures[wi, sl].numpy()
    x_pts  = gt_xy[0]

    le_idx = int(np.argmin(x_pts))
    upper  = np.zeros(len(x_pts), dtype=bool)
    upper[:le_idx + 1] = True
    lower  = ~upper

    h_gt_upper = h_gt_lower = h_rec_upper = h_rec_lower = None
    for mask, ls, surface in [(upper, "-", "upper"), (lower, "--", "lower")]:
        if mask.sum() == 0:
            continue
        idx = np.argsort(x_pts[mask])
        lbl_gt  = f"Ground truth ({surface})"
        lbl_rec = f"{pred_label} ({surface})"
        h_gt_s,  = ax_cp.plot(x_pts[mask][idx], gt_cp[mask][idx],
                               color=COL_GT,    lw=1.0, linestyle=ls, label=lbl_gt)
        h_rec_s, = ax_cp.plot(x_pts[mask][idx], rec_cp[mask][idx],
                               color=COL_RECON, lw=1.0, linestyle=ls, label=lbl_rec)
        if surface == "upper":
            h_gt_upper, h_rec_upper = h_gt_s, h_rec_s
        else:
            h_gt_lower, h_rec_lower = h_gt_s, h_rec_s

    ax_cp.set_xlabel("x/c", fontsize=8)
    if col == 0:
        ax_cp.set_ylabel("Cp", fontsize=8)

    return [h_gt, h_rec, h_gt_upper, h_gt_lower, h_rec_upper, h_rec_lower]


def _savefig(fig, save_stem):
    for ext in ("pdf", "png"):
        path = f"{save_stem}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="results/lvae_3d/lvae_3d_v29_reproducing_best.pth")
    p.add_argument("--bae_checkpoint", default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--bae_latent_dim", type=int, default=128)
    p.add_argument("--lae_latent_dim", type=int, default=64)
    p.add_argument("--n_spans",        type=int, default=15)
    p.add_argument("--wing_indices",   type=int, nargs="+", default=[1, 0, 3],
                   help="Default picks one wing per flow regime: index 1 "
                        "(subsonic), 0 (transonic), 3 (supersonic), labelled "
                        "(a), (b), (c) in that order.")
    p.add_argument("--slice_indices",  type=int, nargs="+", default=None,
                   help="Defaults to all available spanwise slices.")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--save_stem",      default="thesis/figures/lvae_recon_shape_cp",
                   help="Each wing is saved as '{save_stem}_a.pdf', '_b.pdf', ...")
    p.add_argument("--slices_per_row",  type=int, default=3,
                   help="Wrap each wing's slices into stacked row-pairs of "
                        "this many columns (e.g. 3 for 6 slices -> two rows "
                        "of 3; 8 for 15 slices -> two rows of 8 and 7).")
    p.add_argument("--full_slices",    action="store_true",
                   help="Use all spanwise slices instead of 6 evenly spaced "
                        "ones (use with --slices_per_row 8 for the appendix).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bae_model = load_bae_3d(args.bae_checkpoint, device, latent_dim=args.bae_latent_dim)
    model     = load_lvae_3d(args.checkpoint, device, bae_model,
                             bae_latent_dim=args.bae_latent_dim,
                             lae_latent_dim=args.lae_latent_dim,
                             n_spans=args.n_spans)

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_dataset = [item for item in list(new_dataset["test"]) if item["final"] == 1]
    n_test       = len(test_dataset)
    print(f"Test wings: {n_test}")

    max_idx = max(args.wing_indices)
    items   = test_dataset[:max_idx + 1]

    gt_airfoils, gt_pressures = [], []
    z_baes, params, te_shifts, le_x, chord = [], [], [], [], []
    eta_values = None
    for item in items:
        z_bae, gt_coords, _aoa, params_scaled, te, pressure, lex, ch = encode_item_3d(
            item, bae_model, model, device, apply_x_norm=True,
        )
        gt_airfoils.append(gt_coords)
        gt_pressures.append(pressure)
        z_baes.append(z_bae)
        params.append(params_scaled)
        te_shifts.append(te)
        le_x.append(lex)
        chord.append(ch)
        if eta_values is None and "transforms" in item:
            eta_values = item["transforms"]  # [n_spans] span positions

    gt_airfoils_t  = torch.stack(gt_airfoils)
    gt_pressures_t = torch.stack(gt_pressures)
    z_baes_t       = torch.stack(z_baes)
    params_t       = torch.cat(params, dim=0)
    te_shifts_t    = torch.stack(te_shifts)
    le_x_t         = torch.stack(le_x)
    chord_t        = torch.stack(chord)

    rec_airfoils_t, _aoas, rec_pressures_t, _perf = reconstruct_batch_3d(
        model, bae_model, z_baes_t, gt_pressures_t, params_t,
        te_shifts_t, le_x_t, chord_t, device,
    )

    n_spans = gt_airfoils_t.shape[1]
    if args.slice_indices is not None:
        slice_indices = args.slice_indices
    elif args.full_slices:
        slice_indices = list(range(n_spans))
    else:
        # 6 evenly spaced slices root -> tip
        slice_indices = list(np.linspace(0, n_spans - 1, 6, dtype=int))

    plot_recon(
        gt_airfoils_t, rec_airfoils_t, gt_pressures_t, rec_pressures_t,
        wing_indices=args.wing_indices, slice_indices=slice_indices,
        save_stem=args.save_stem, slices_per_row=args.slices_per_row,
        eta_values=eta_values,
    )


if __name__ == "__main__":
    main()
