"""
LVAE latent space figures.

Usage:
    .venv/bin/python plot_latent.py --component std
    .venv/bin/python plot_latent.py --component dims [--top_n 10]
    .venv/bin/python plot_latent.py --component traversal
    .venv/bin/python plot_latent.py --component correlations
    .venv/bin/python plot_latent.py --component w50
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = "thesis/figures"
os.makedirs(OUT, exist_ok=True)

BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"
_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

C_TRAIN = "#4477AA"
C_GEN   = "#EE6677"


def _load_models(device="cpu"):
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
    bae  = load_bae_3d(BAE_CKP, device)
    lvae = load_lvae_3d(LVAE_CKP, device, bae)
    return bae, lvae


def _train_items(seed=42):
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    ds = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=seed)
    return [it for it in ds["train"] if it["final"] == 1], ds


# ── std ───────────────────────────────────────────────────────────────────────

def plot_std():
    from matplotlib.patches import Patch
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d

    TAU = 0.02
    device = "cpu"
    bae, lvae = _load_models(device)
    items, _ = _train_items()
    print(f"Encoding {len(items)} train wings...")

    ws = []
    for item in items:
        z_bae, _, _, params_s, _, pressure, _, _ = encode_item_3d(
            item, bae, lvae, device, apply_x_norm=True,
        )
        with torch.no_grad():
            w = lvae.encoder(
                z_bae.unsqueeze(0).to(device),
                pressure.unsqueeze(0).to(device),
                params_s.to(device),
            )
        ws.append(w.squeeze(0).cpu())

    ws   = torch.stack(ws)
    stds = ws.std(dim=0).numpy()
    mask  = lvae.active_latent_mask.cpu().numpy()
    order = np.argsort(stds)[::-1]
    stds_s = stds[order]
    mask_s = mask[order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(stds_s) + 1), stds_s, color="#4477AA", width=0.8, edgecolor="none")
    ax.axhline(TAU, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.5, len(stds_s) + 0.5)
    ax.set_xlabel(r"Latent dimension (sorted by $\sigma$, descending)", fontsize=13)
    ax.set_ylabel(r"Standard deviation $\sigma_i$", fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    legend_elements = [
        Patch(facecolor="#4477AA", label=f"All dims (64 total, 20 active)"),
        plt.Line2D([0], [0], color="#333333", linestyle="--", lw=1.0, label=r"$\tau = 0.02$"),
    ]
    ax.legend(handles=legend_elements, fontsize=12, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(axis="y", direction="out", which="both", right=False)
    ax.tick_params(axis="x", which="both", direction="out", top=False)
    fig.tight_layout()

    for path in (f"results/lvae_3d_evaluation/latent_std_lvae_3d_v29_best",
                 f"{OUT}/latent_std_lvae_3d_v29_best"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=150, bbox_inches="tight")
            print(f"Saved: {path}.{ext}")
    plt.close()


# ── dims ──────────────────────────────────────────────────────────────────────

def plot_dims(top_n=None):
    import pickle
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
    from engiopt.lvae.plot_latent_tsne import _normalise_coords, collect_w_latents
    from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
    from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser as MLPDenoiserW
    from engiopt.ddm.ddm_w.train_ddm_w_3d import Config as ConfigW, build_sampler as build_sampler_w
    from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d as precompute_ddmw
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    from engiopt.data_processing.utils import scaler

    DDMW_CKP = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"
    N_GEN    = 200
    DEVICE   = "cpu"

    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans"})

    def _disable_usetex():
        plt.rcParams.update({"text.usetex": False})

    def load_ddm_w(bae_model, lvae_model, device):
        ckpt = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
        cfg  = ConfigW()
        cfg.bae_checkpoint  = BAE_CKP
        cfg.lvae_checkpoint = LVAE_CKP
        cfg.device          = device
        sampler = build_sampler_w(cfg)
        saved   = ckpt["denoiser"]
        if isinstance(saved, MLPDenoiserW):
            denoiser = saved
        else:
            denoiser = MLPDenoiserW(w_dim=cfg.lae_latent_dim, c_dim=cfg.c_dim)
            denoiser.load_state_dict(saved)
        ddm_w = DDM_W3D(
            denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
            sampler=sampler, w_dim=cfg.lae_latent_dim, c_dim=cfg.c_dim,
            w_pressure=ckpt.get("w_pressure", 1.0),
            w_aoa=ckpt.get("w_aoa", 1.0),
            lvae_params_dim=cfg.c_dim,
            params_mean_std=ckpt.get("params_mean_std"),
            aoas_mean_std=ckpt.get("aoas_mean_std"),
        )
        ddm_w.w_mean     = ckpt.get("w_mean")
        ddm_w.w_std      = ckpt.get("w_std")
        ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
        ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
        ddm_w.denoiser.to(device).eval()
        return ddm_w, ckpt, cfg

    def get_generated_w(ddm_w, bae_model, lvae_model, ckpt, cfg, device, n=N_GEN):
        dataset         = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
        all_test        = list(dataset["test"])
        initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
        test_final      = [it for it in all_test if it["final"] == 1]
        sc_params = scaler(ckpt["params_mean_std"]) if ckpt.get("params_mean_std") is not None else None
        sc_aoas   = scaler(ckpt["aoas_mean_std"])   if ckpt.get("aoas_mean_std")   is not None else None
        _, _, _, _, _, w_inits, params_norm, _ = precompute_ddmw(
            test_final, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, device,
        )
        n = min(n, len(w_inits))
        with torch.no_grad():
            gen_coords, _, gen_pressures, _, _, _ = ddm_w.generate(
                w_init=w_inits[:n].to(device),
                params=params_norm[:n].to(device),
                device=device,
            )
        test_final_n = test_final[:n]
        flow_params  = []
        for item in test_final_n:
            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_params.append(torch.tensor(flow, dtype=torch.float32))
        bae_model.eval()
        lvae_model.encoder.eval()
        w_list = []
        with torch.no_grad():
            for i, item in enumerate(test_final_n):
                x_wing   = gen_coords[i].unsqueeze(0).to(device)
                z_bae    = bae_model.encode(x_wing)
                params   = flow_params[i].unsqueeze(0).to(device)
                params_s = lvae_model.scaler_params.transform(params)
                pressure = torch.tensor(item["coef_pressure"], dtype=torch.float32).unsqueeze(0).to(device)
                w = lvae_model.encoder(z_bae, pressure, params_s)
                w_list.append(w.squeeze(0).cpu())
        return torch.stack(w_list).numpy()

    def plot_lvae_pairs(train_w, gen_w, n_dims, save_path, dim_labels=None,
                        training_only=False, legend_y=0.07):
        _disable_usetex()
        if dim_labels is None:
            dim_labels = [f"w{i+1}" for i in range(n_dims)]
        from matplotlib.lines import Line2D
        fig, axes = plt.subplots(n_dims, n_dims,
                                  figsize=(2.3 * n_dims, 2.3 * n_dims),
                                  sharex=False, sharey=False)
        fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.10)
        for row in range(n_dims):
            for col in range(n_dims):
                ax = axes[row, col]
                if row == col:
                    lo = train_w[:, col].min() if training_only else min(train_w[:, col].min(), gen_w[:, col].min())
                    hi = train_w[:, col].max() if training_only else max(train_w[:, col].max(), gen_w[:, col].max())
                    if hi == lo: hi = lo + 1e-6
                    bins = np.linspace(lo, hi, 31)
                    ax.hist(train_w[:, col], bins=bins, color=C_TRAIN, alpha=0.6,
                            density=True, histtype="stepfilled")
                    if not training_only:
                        ax.hist(gen_w[:, col], bins=bins, color=C_GEN, alpha=0.7,
                                density=True, histtype="step", linewidth=1.2)
                elif row > col:
                    ax.scatter(train_w[:, col], train_w[:, row],
                               s=4, alpha=0.35, color=C_TRAIN, rasterized=True)
                    if not training_only:
                        ax.scatter(gen_w[:, col], gen_w[:, row],
                                   s=8, alpha=0.65, color=C_GEN, marker="x", rasterized=True)
                else:
                    ax.set_visible(False)
                    ax.set_frame_on(False)
                    continue
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                if col == 0 and row > 0:
                    ax.set_ylabel(dim_labels[row], fontsize=24, labelpad=4)
                if row == n_dims - 1:
                    ax.set_xlabel(dim_labels[col], fontsize=24, labelpad=4)
        if not training_only:
            handles = [
                Line2D([0], [0], color=C_TRAIN, lw=0, marker="o", markersize=9, alpha=0.7, label="Training set"),
                Line2D([0], [0], color=C_GEN,   lw=0, marker="x", markersize=10, label="Generated (DDM-W)"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=22,
                       bbox_to_anchor=(0.5, legend_y), frameon=False)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        fig.savefig(save_path.replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    torch.manual_seed(42)
    np.random.seed(42)
    bae, lvae = _load_models(DEVICE)
    items, ds = _train_items()
    print(f"Training wings: {len(items)}")
    train_w = collect_w_latents(lvae, bae, items, DEVICE)

    ddm_w, ckpt, cfg = load_ddm_w(bae, lvae, DEVICE)
    gen_w = get_generated_w(ddm_w, bae, lvae, ckpt, cfg, DEVICE, n=N_GEN)

    train_std  = train_w.std(axis=0)
    active_idx = np.where(train_std >= 0.02)[0]
    active_idx = active_idx[np.argsort(train_std[active_idx])[::-1]]
    print(f"Active dimensions ({len(active_idx)}): {active_idx.tolist()}")

    plot_lvae_pairs(
        train_w[:, active_idx], gen_w[:, active_idx],
        n_dims=len(active_idx),
        dim_labels=[f"w{i+1}" for i in active_idx],
        save_path=os.path.join(OUT, "lvae_latent_dims.pdf"),
    )

    if top_n is not None:
        n = min(top_n, len(active_idx))
        plot_lvae_pairs(
            train_w[:, active_idx[:n]], gen_w[:, active_idx[:n]],
            n_dims=n, dim_labels=[f"w{i+1}" for i in active_idx[:n]],
            legend_y=0.01,
            save_path=os.path.join(OUT, f"lvae_latent_dims_top{n}.pdf"),
        )
        plot_lvae_pairs(
            train_w[:, active_idx[:n]], gen_w[:, active_idx[:n]],
            n_dims=n, dim_labels=[f"w{i+1}" for i in active_idx[:n]],
            save_path=os.path.join(OUT, f"lvae_latent_dims_top{n}_train_only.pdf"),
            training_only=True,
        )


# ── traversal ─────────────────────────────────────────────────────────────────

def plot_traversal():
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d
    from engiopt.lvae.plot_latent_tsne import collect_w_latents
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

    N_STEPS = 7
    N_SIGMA = 3.0
    DEVICE  = "cpu"

    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif",
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dark_div", ["#1a4f8a", "#6a3d9a", "#c0392b"]
    )

    bae, lvae = _load_models(DEVICE)
    dataset   = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    train_items = [it for it in dataset["train"] if it["final"] == 1]
    test_items  = [it for it in dataset["test"]  if it["final"] == 1]

    train_w   = collect_w_latents(lvae, bae, train_items, DEVICE)
    train_std = train_w.std(axis=0)
    active_idx = np.where(train_std >= 0.02)[0]
    active_idx = active_idx[np.argsort(train_std[active_idx])[::-1]]
    print(f"Active dims ({len(active_idx)}): {active_idx.tolist()}")

    idx_top1   = active_idx[0]
    idx_bottom = active_idx[-1]
    col_top1   = 0
    col_bottom = len(active_idx) - 1
    train_active = train_w[:, active_idx]

    ref_item = next(it for it in test_items if 0.8 <= it["mach"] < 1.0)
    print(f"Reference wing: Mach={ref_item['mach']:.3f}")
    z_bae, _, _, params_scaled, _, gt_pressure, _, _ = encode_item_3d(
        ref_item, bae, lvae, DEVICE, apply_x_norm=True,
    )
    with torch.no_grad():
        w_ref = lvae.encoder(
            z_bae.unsqueeze(0).to(DEVICE),
            gt_pressure.unsqueeze(0).to(DEVICE),
            params_scaled.to(DEVICE),
        )
    w_ref_np = w_ref.squeeze(0).cpu().numpy()

    dim_std_top1 = float(train_std[idx_top1])
    vals = np.linspace(-N_SIGMA * dim_std_top1, N_SIGMA * dim_std_top1, N_STEPS)
    traversal_w = np.tile(w_ref_np, (N_STEPS, 1))
    traversal_w[:, idx_top1] = vals
    traversal_active = traversal_w[:, active_idx]
    sigma_vals = np.linspace(-N_SIGMA, N_SIGMA, N_STEPS)

    bottom_label = idx_bottom + 1
    step_colors  = CMAP(np.linspace(0, 1, N_STEPS))

    # Figure A: scatter + traversal path
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(train_active[:, col_top1], train_active[:, col_bottom],
               s=6, alpha=0.35, color=C_TRAIN, label="Training set", rasterized=True)
    for k in range(N_STEPS - 1):
        ax.plot(traversal_active[k:k+2, col_top1], traversal_active[k:k+2, col_bottom],
                color=step_colors[k], linewidth=2.0, zorder=3)
    sc = ax.scatter(traversal_active[:, col_top1], traversal_active[:, col_bottom],
                    c=np.linspace(0, 1, N_STEPS), cmap=CMAP,
                    s=60, zorder=4, edgecolors="k", linewidths=0.5, label="Traversal steps")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels([f"$-{N_SIGMA:.0f}\\sigma$", "0", f"$+{N_SIGMA:.0f}\\sigma$"])
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(f"$w_{{45}}$")
    ax.set_ylabel(f"$w_{{{bottom_label}}}$")
    ax.legend(fontsize=11, markerscale=1.5, frameon=False)
    fig.tight_layout()
    path_a = os.path.join(OUT, "traversal_manifold_overlay.pdf")
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    fig.savefig(path_a.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_a}")

    # Figure B: NN distance
    diffs    = train_active[np.newaxis, :, :] - traversal_active[:, np.newaxis, :]
    nn_dists = np.sqrt((diffs ** 2).sum(axis=-1)).min(axis=1)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for k in range(N_STEPS - 1):
        ax.plot(sigma_vals[k:k+2], nn_dists[k:k+2], color=step_colors[k], linewidth=2.0)
    ax.scatter(sigma_vals, nn_dists, c=np.linspace(0, 1, N_STEPS), cmap=CMAP,
               s=60, zorder=3, edgecolors="k", linewidths=0.5)
    ax.set_xlabel(f"$w_{{45}}$ (multiples of $\\sigma$)")
    ax.set_ylabel("Distance to nearest\ntraining point")
    ax.set_xticks(sigma_vals)
    ax.set_xticklabels([f"{v:.1f}$\\sigma$" for v in sigma_vals], fontsize=10)
    fig.tight_layout()
    path_b = os.path.join(OUT, "traversal_nn_distance.pdf")
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    fig.savefig(path_b.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_b}")


# ── correlations ──────────────────────────────────────────────────────────────

def plot_correlations():
    from scipy.stats import pearsonr
    import pandas as pd
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d
    from engiopt.lvae.plot_latent_traversal import compute_dim_stds, pick_traversal_dims
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bae    = load_bae_3d(BAE_CKP, device, latent_dim=128)
    lvae   = load_lvae_3d(LVAE_CKP, device, bae, bae_latent_dim=128, lae_latent_dim=64, n_spans=15)

    dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    test_dataset = [it for it in list(dataset["test"]) if it["final"] == 1]
    print(f"Test wings: {len(test_dataset)}")

    scalars_df = pd.read_pickle(_SCALARS_PKL)
    SCALAR_COLS = [
        "alpha", "cl", "cd",
        "case_mach", "case_reynolds", "case_cl_target", "case_alpha",
        "case_sweep_value", "case_twist_tip", "case_base_span_scaling_value",
        "case_initial_spanwise_taper_scaling_value", "case_initial_thickness_taper_scaling_value",
        "case_volume_ratio_min",
        "case_chord_scaling_value0", "case_chord_scaling_value3", "case_chord_scaling_value6",
        "case_thickness_taper_value0", "case_thickness_taper_value3", "case_thickness_taper_value6",
        "case_twist_value0", "case_twist_value3", "case_twist_value6",
        "case_dihedral_value0", "case_dihedral_value3", "case_dihedral_value6",
    ]
    PRETTY = {
        "alpha": "AoA (solved)", "cl": "CL (solved)", "cd": "CD (solved)",
        "case_mach": "Mach", "case_reynolds": "Reynolds", "case_cl_target": "CL target",
        "case_alpha": "AoA (target)", "case_sweep_value": "Sweep", "case_twist_tip": "Tip twist",
        "case_base_span_scaling_value": "Span scale",
        "case_initial_spanwise_taper_scaling_value": "Spanwise taper",
        "case_initial_thickness_taper_scaling_value": "Thickness taper",
        "case_volume_ratio_min": "Min volume ratio",
        "case_chord_scaling_value0": "Chord scale (root)", "case_chord_scaling_value3": "Chord scale (mid)",
        "case_chord_scaling_value6": "Chord scale (tip)",
        "case_thickness_taper_value0": "Thickness taper (root)",
        "case_thickness_taper_value3": "Thickness taper (mid)",
        "case_thickness_taper_value6": "Thickness taper (tip)",
        "case_twist_value0": "Twist (root)", "case_twist_value3": "Twist (mid)",
        "case_twist_value6": "Twist (tip)",
        "case_dihedral_value0": "Dihedral (root)", "case_dihedral_value3": "Dihedral (mid)",
        "case_dihedral_value6": "Dihedral (tip)",
    }

    z_baes_list, params_list, pressures_list, scalar_rows = [], [], [], []
    for item in test_dataset:
        z_bae, _, _, params_scaled, _, pressure, _, _ = encode_item_3d(
            item, bae, lvae, device, apply_x_norm=True,
        )
        z_baes_list.append(z_bae)
        params_list.append(params_scaled.squeeze(0))
        pressures_list.append(pressure)
        case_rows = scalars_df[scalars_df["case_num"] == item["case_num"]]
        scalar_rows.append(case_rows.iloc[-1] if len(case_rows) > 0 else None)

    z_baes_t    = torch.stack(z_baes_list)
    params_t    = torch.stack(params_list)
    pressures_t = torch.stack(pressures_list)

    dim_stds, all_w = compute_dim_stds(lvae, z_baes_t, pressures_t, params_t, device)
    active_mask = lvae.active_latent_mask.cpu()
    top_k = 3
    top_indices = pick_traversal_dims(dim_stds, active_mask)[:top_k]

    valid_idx  = [i for i, r in enumerate(scalar_rows) if r is not None]
    scalar_mat = np.array([[scalar_rows[i][c] for c in SCALAR_COLS] for i in valid_idx], dtype=float)
    w_mat      = all_w[valid_idx].numpy()

    n_show  = 10
    colours = ["#c0392b", "#2980b9", "#27ae60"]

    fig, axes = plt.subplots(1, top_k, figsize=(5.5 * top_k, 6), sharey=False)
    if top_k == 1:
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
        ranked = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:n_show]
        labels = [PRETTY.get(c, c) for c, _ in ranked]
        vals   = [r for _, r in ranked]
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
    out_path = os.path.join(OUT, "latent_correlations.pdf")
    for ext in ("pdf", "png"):
        fig.savefig(out_path.replace(".pdf", f".{ext}"), dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path.replace('.pdf', f'.{ext}')}")
    plt.close()


# ── w50 ───────────────────────────────────────────────────────────────────────

def plot_w50():
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

    DIM = 49
    REGIME_COLORS = {"subsonic": "#7BAFD4", "transonic": "#F0A868", "supersonic": "#D96B5A"}
    REGIME_LABELS = {
        "subsonic":   "Subsonic\n($M < 0.8$)",
        "transonic":  "Transonic\n($0.8 \leq M < 1.0$)",
        "supersonic": "Supersonic\n($M \geq 1.0$)",
    }
    REGIME_ORDER = ["subsonic", "transonic", "supersonic"]

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    bae_model  = load_bae_3d(BAE_CKP, device)
    lvae_model = load_lvae_3d(LVAE_CKP, device, bae_model=bae_model)
    lvae_model.encoder.eval()

    dataset    = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    train_items = [item for item in dataset["train"] if item["final"] == 1]
    print(f"Training wings: {len(train_items)}")

    ws, machs = [], []
    with torch.no_grad():
        for i, item in enumerate(train_items):
            if i % 100 == 0:
                print(f"  {i}/{len(train_items)}")
            z_bae, _, _, params_scaled, _, pressure, _, _ = encode_item_3d(
                item, bae_model, lvae_model, device)
            w = lvae_model.encoder(
                z_bae.unsqueeze(0).to(device),
                pressure.unsqueeze(0).to(device),
                params_scaled.to(device),
            )
            w = lvae_model._apply_mask(w)
            ws.append(w.squeeze(0).cpu().numpy())
            machs.append(item["mach"])

    ws    = np.array(ws)
    machs = np.array(machs)

    regime_masks = {
        "subsonic":   machs < 0.8,
        "transonic":  (machs >= 0.8) & (machs < 1.0),
        "supersonic": machs >= 1.0,
    }
    all_vals    = ws[:, DIM]
    global_min, global_max = all_vals.min(), all_vals.max()
    bin_width   = (global_max - global_min) / 30
    shared_bins = np.arange(global_min, global_max + bin_width, bin_width)

    for reg in REGIME_ORDER:
        mask  = regime_masks[reg]
        vals  = ws[mask, DIM]
        color = REGIME_COLORS[reg]
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        ax.hist(vals, bins=shared_bins, color=color, alpha=0.85, edgecolor="none")
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        out = os.path.join(OUT, f"w50_regime_hist_{reg}.pdf")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component", required=True,
                   choices=["std", "dims", "traversal", "correlations", "w50"])
    p.add_argument("--top_n", type=int, default=None,
                   help="For --component dims: also save a top-N pair plot")
    args = p.parse_args()

    if args.component == "std":
        plot_std()
    elif args.component == "dims":
        plot_dims(top_n=args.top_n)
    elif args.component == "traversal":
        plot_traversal()
    elif args.component == "correlations":
        plot_correlations()
    elif args.component == "w50":
        plot_w50()


if __name__ == "__main__":
    main()
