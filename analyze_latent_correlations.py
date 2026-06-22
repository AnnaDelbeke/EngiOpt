"""
Compute Pearson correlations between the top-3 highest-variance LVAE latent
dimensions and all available physical / aerodynamic scalars.

Prints a ranked correlation table per dim and saves a bar-chart figure.
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import encode_item_3d, load_bae_3d, load_lvae_3d
from engiopt.lvae.plot_latent_traversal import (
    compute_dim_stds, pick_traversal_dims,
)

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
BAE_CKP      = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP     = "results/lvae_3d/lvae_3d_v29_best.pth"

# Scalars to include — drop identifiers, flags, and mesh internals
SCALAR_COLS = [
    "alpha", "cl", "cd",
    "case_mach", "case_reynolds", "case_cl_target", "case_alpha",
    "case_sweep_value", "case_twist_tip",
    "case_base_span_scaling_value",
    "case_initial_spanwise_taper_scaling_value",
    "case_initial_thickness_taper_scaling_value",
    "case_volume_ratio_min",
    "case_chord_scaling_value0", "case_chord_scaling_value3", "case_chord_scaling_value6",
    "case_thickness_taper_value0", "case_thickness_taper_value3", "case_thickness_taper_value6",
    "case_twist_value0", "case_twist_value3", "case_twist_value6",
    "case_dihedral_value0", "case_dihedral_value3", "case_dihedral_value6",
]

PRETTY = {
    "alpha": "AoA (solved)",
    "cl": "CL (solved)",
    "cd": "CD (solved)",
    "case_mach": "Mach",
    "case_reynolds": "Reynolds",
    "case_cl_target": "CL target",
    "case_alpha": "AoA (target)",
    "case_sweep_value": "Sweep",
    "case_twist_tip": "Tip twist",
    "case_base_span_scaling_value": "Span scale",
    "case_initial_spanwise_taper_scaling_value": "Spanwise taper",
    "case_initial_thickness_taper_scaling_value": "Thickness taper",
    "case_volume_ratio_min": "Min volume ratio",
    "case_chord_scaling_value0": "Chord scale (root)",
    "case_chord_scaling_value3": "Chord scale (mid)",
    "case_chord_scaling_value6": "Chord scale (tip)",
    "case_thickness_taper_value0": "Thickness taper (root)",
    "case_thickness_taper_value3": "Thickness taper (mid)",
    "case_thickness_taper_value6": "Thickness taper (tip)",
    "case_twist_value0": "Twist (root)",
    "case_twist_value3": "Twist (mid)",
    "case_twist_value6": "Twist (tip)",
    "case_dihedral_value0": "Dihedral (root)",
    "case_dihedral_value3": "Dihedral (mid)",
    "case_dihedral_value6": "Dihedral (tip)",
}

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=LVAE_CKP)
    p.add_argument("--bae_checkpoint", default=BAE_CKP)
    p.add_argument("--top_k",          type=int, default=3)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--out",            default="thesis/figures/latent_correlations.pdf")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bae_model = load_bae_3d(args.bae_checkpoint, device, latent_dim=128)
    model     = load_lvae_3d(args.checkpoint, device, bae_model,
                             bae_latent_dim=128, lae_latent_dim=64, n_spans=15)

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_dataset = [it for it in list(new_dataset["test"]) if it["final"] == 1]
    print(f"Test wings: {len(test_dataset)}")

    # ── encode all test wings ────────────────────────────────────────────────
    import pandas as pd
    scalars_df = pd.read_pickle(_SCALARS_PKL)

    z_baes_list, params_list, pressures_list = [], [], []
    scalar_rows = []

    for item in test_dataset:
        z_bae, _gt, _aoa, params_scaled, _te, pressure, _le, _ch = encode_item_3d(
            item, bae_model, model, device, apply_x_norm=True,
        )
        z_baes_list.append(z_bae)
        params_list.append(params_scaled.squeeze(0))
        pressures_list.append(pressure)

        # match scalar row by case_num (final iteration)
        case_rows = scalars_df[scalars_df["case_num"] == item["case_num"]]
        if len(case_rows) > 0:
            scalar_rows.append(case_rows.iloc[-1])  # last iter = final
        else:
            scalar_rows.append(None)

    z_baes_t    = torch.stack(z_baes_list)
    params_t    = torch.stack(params_list)
    pressures_t = torch.stack(pressures_list)

    # ── compute per-dim stds and pick top-k dims ─────────────────────────────
    dim_stds, all_w = compute_dim_stds(model, z_baes_t, pressures_t, params_t, device)
    active_mask = model.active_latent_mask.cpu()
    top_indices = pick_traversal_dims(dim_stds, active_mask)[:args.top_k]
    print(f"Top-{args.top_k} dims: {top_indices}")

    # ── build scalar matrix (keep only rows with valid scalars) ──────────────
    valid_mask = [r is not None for r in scalar_rows]
    valid_idx  = [i for i, v in enumerate(valid_mask) if v]
    scalar_mat = np.array([
        [scalar_rows[i][c] for c in SCALAR_COLS]
        for i in valid_idx
    ], dtype=float)                                    # [N, n_scalars]
    w_mat = all_w[valid_idx].numpy()                   # [N, latent_dim]

    print(f"Valid wings with scalars: {len(valid_idx)}")

    # ── all-data figure (top-10 from all scalars, no regime split) ──────────
    from scipy.stats import pearsonr

    n_show   = 10
    n_dims   = len(top_indices)
    colours  = ["#c0392b", "#2980b9", "#27ae60"]

    fig, axes = plt.subplots(1, n_dims, figsize=(5.5 * n_dims, 6), sharey=False)
    if n_dims == 1:
        axes = [axes]

    for ax, dim_idx, colour in zip(axes, top_indices, colours):
        w_vals  = w_mat[:, dim_idx]
        dim_std = float(np.std(w_mat[:, dim_idx]))
        corrs   = {}
        for col in SCALAR_COLS:
            j = SCALAR_COLS.index(col)
            s = scalar_mat[:, j]
            if np.std(s) < 1e-10 or np.std(w_vals) < 1e-10:
                corrs[col] = 0.0
            else:
                r, _ = pearsonr(w_vals, s)
                corrs[col] = r

        ranked      = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:n_show]
        labels      = [PRETTY.get(c, c) for c, _ in ranked]
        vals        = [r for _, r in ranked]
        bar_colours = [colour if v >= 0 else "#aab4c8" for v in vals]

        ax.barh(range(len(ranked)), vals, color=bar_colours, alpha=0.85)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Pearson r")
        ax.set_title(f"w{dim_idx}  ($\\sigma$ = {dim_std:.2f})")
        ax.invert_yaxis()
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.out.replace(".pdf", f".{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # ── split into regimes ───────────────────────────────────────────────────

    mach_col = SCALAR_COLS.index("case_mach")
    mach_vals = scalar_mat[:, mach_col]
    regimes = {
        "Subsonic\n(Mach < 0.75)":       mach_vals < 0.75,
        "Transonic\n(0.75 ≤ Mach < 0.95)": (mach_vals >= 0.75) & (mach_vals < 0.95),
        "Supersonic\n(Mach ≥ 0.95)":     mach_vals >= 0.95,
    }

    # Geometric / design scalars only (exclude Mach, Re, CL, CD, AoA)
    GEOM_COLS = [c for c in SCALAR_COLS if c not in (
        "case_mach", "case_reynolds", "case_cl_target", "case_alpha",
        "alpha", "cl", "cd",
    )]

    n_show  = 8
    n_dims  = len(top_indices)
    n_reg   = len(regimes)

    # one figure per regime: rows=dims, cols=top-n_show bars
    for reg_label, reg_mask in regimes.items():
        n_reg_wings = reg_mask.sum()
        reg_name = reg_label.split("\n")[0].replace(" ", "_").lower()
        print(f"\n{'='*60}")
        print(f"Regime: {reg_label.replace(chr(10),' ')}  (n={n_reg_wings})")

        fig, axes = plt.subplots(1, n_dims, figsize=(5.5 * n_dims, 5), sharey=False)
        if n_dims == 1:
            axes = [axes]

        colours = ["#c0392b", "#2980b9", "#27ae60"]

        for ax, dim_idx, colour in zip(axes, top_indices, colours):
            w_vals = w_mat[reg_mask, dim_idx]
            corrs  = {}
            for col in GEOM_COLS:
                j = SCALAR_COLS.index(col)
                s = scalar_mat[reg_mask, j]
                if np.std(s) < 1e-10 or np.std(w_vals) < 1e-10:
                    corrs[col] = 0.0
                else:
                    r, _ = pearsonr(w_vals, s)
                    corrs[col] = r

            ranked = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:n_show]
            print(f"\n  w{dim_idx}:")
            for col, r in ranked:
                print(f"    {PRETTY.get(col, col):35s}  r = {r:+.3f}")

            labels      = [PRETTY.get(c, c) for c, _ in ranked]
            vals        = [r for _, r in ranked]
            bar_colours = [colour if v >= 0 else "#aab4c8" for v in vals]

            ax.barh(range(len(ranked)), vals, color=bar_colours, alpha=0.85)
            ax.set_yticks(range(len(ranked)))
            ax.set_yticklabels(labels)
            ax.axvline(0, color="black", lw=0.7)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("Pearson r")
            ax.set_title(f"w{dim_idx}  (n={n_reg_wings})")
            ax.invert_yaxis()
            ax.grid(axis="x", linewidth=0.4, alpha=0.5)

        fig.tight_layout()

        stem = args.out.replace(".pdf", f"_{reg_name}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        for ext in ("pdf", "png"):
            path = f"{stem}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close()


if __name__ == "__main__":
    main()
