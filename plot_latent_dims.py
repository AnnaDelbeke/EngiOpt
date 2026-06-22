"""
Direct latent-space scatter plots (no t-SNE).

LVAE plot:  20×20 pair-plot grid of the first 20 LVAE w-dimensions.
            Training set (encoded) vs. DDM-W generated samples.

PCA plot:   3-panel figure: PC1 vs PC2, PC1 vs PC3, PC2 vs PC3.
            Training set (BAE latents projected through PCA) vs. DDM-PCA generated.

Best checkpoints used:
  BAE:     results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt
  LVAE:    results/lvae_3d/lvae_3d_v29_best.pth
  DDM-W:   results/ddm_w_3d/ddm_w_3d_v29_best.pth
  DDM-PCA: results/ddm_pca_3d/ddm_pca_3d_v2_best.pth  (+  _pca.pkl)

Usage:
    python plot_latent_dims.py
"""

import sys
import os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

import pickle
import numpy as np
import torch

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans"})

def _disable_usetex():
    plt.rcParams.update({"text.usetex": False})

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.train_lvae_3d import LVAE3D, LAEEncoderJoint3D
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
from engiopt.lvae.plot_latent_tsne import _normalise_coords, collect_w_latents
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser as MLPDenoiserW
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config as ConfigW, build_sampler as build_sampler_w
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d as precompute_ddmw
from engiopt.ddm.ddm_pca.ddm_pca_3d import DDM_PCA_3D, MLPDenoiser as MLPDenoiserPCA
from engiopt.ddm.ddm_pca.train_ddm_pca_3d import Config as ConfigPCA, build_sampler as build_sampler_pca
from engiopt.ddm.ddm_pca.evaluate_ddm_pca_3d import precompute_test_3d as precompute_ddmpca
from engiopt.data_processing.utils import scaler

# ── paths ──────────────────────────────────────────────────────────────────
BAE_CKP     = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP    = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP    = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"
DDMPCA_CKP  = "results/ddm_pca_3d/ddm_pca_3d_v2_best.pth"
PCA_PKL     = "results/ddm_pca_3d/ddm_pca_3d_v2_pca.pkl"

OUT_DIR = "thesis/figures"
DEVICE  = "cpu"
N_GEN   = 200   # generated samples to overlay

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

C_TRAIN = "#4477AA"   # blue — training set
C_GEN   = "#EE6677"   # red  — generated


# ---------------------------------------------------------------------------
# DDM-W: load model and generate w-vectors
# ---------------------------------------------------------------------------

def load_ddm_w(bae_model, lvae_model, device):
    ckpt = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    cfg  = ConfigW()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = device
    sampler = build_sampler_w(cfg)

    saved = ckpt["denoiser"]
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
    return ddm_w


def get_generated_w(ddm_w, bae_model, lvae_model, device, n=N_GEN):
    dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    all_test     = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_final      = [it for it in all_test if it["final"] == 1]

    ckpt = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
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

    # Re-encode generated coords through the LVAE encoder so gen_w is in the
    # exact same space as the train_w from collect_w_latents (raw encoder output).
    test_final_n = test_final[:n]
    flow_params = []
    for item in test_final_n:
        flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        flow_params.append(torch.tensor(flow, dtype=torch.float32))

    bae_model.eval()
    lvae_model.encoder.eval()
    w_list = []
    with torch.no_grad():
        for i, item in enumerate(test_final_n):
            x_wing  = gen_coords[i].unsqueeze(0).to(device)   # [1, S, 2, 192]
            z_bae   = bae_model.encode(x_wing)                 # [1, bae_latent_dim]
            params  = flow_params[i].unsqueeze(0).to(device)
            params_s = lvae_model.scaler_params.transform(params)
            pressure = torch.tensor(
                item["coef_pressure"], dtype=torch.float32
            ).unsqueeze(0).to(device)
            w = lvae_model.encoder(z_bae, pressure, params_s)
            w_list.append(w.squeeze(0).cpu())

    return torch.stack(w_list).numpy()   # [n, lae_latent_dim]


# ---------------------------------------------------------------------------
# DDM-PCA: load model and generate PCA z-vectors
# ---------------------------------------------------------------------------

def load_ddm_pca(bae_model, pca, device):
    ckpt = torch.load(DDMPCA_CKP, map_location="cpu", weights_only=False)
    cfg  = ConfigPCA()
    z_dim = ckpt.get("z_dim", cfg.n_components)

    denoiser = MLPDenoiserPCA(z_dim=z_dim, c_dim=cfg.c_dim).to(device)
    model = DDM_PCA_3D(
        denoiser=denoiser, pca=pca, bae_model=bae_model,
        sampler=build_sampler_pca(cfg), z_dim=z_dim, c_dim=cfg.c_dim,
        w_aoa=ckpt.get("w_aoa", 1.0),
        params_mean_std=ckpt.get("params_mean_std"),
        aoas_mean_std=ckpt.get("aoas_mean_std"),
    )
    model.load(DDMPCA_CKP, train_mode=False)
    model.denoiser.to(device)
    return model


def get_generated_pca_z(ddm_pca_model, bae_model, pca, device, n=N_GEN):
    dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    all_test     = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_final      = [it for it in all_test if it["final"] == 1]

    _, _, _, z_inits, params_norm = precompute_ddmpca(
        test_final, initial_by_case, bae_model, pca,
        ddm_pca_model.z_mean, ddm_pca_model.z_std,
        ddm_pca_model.scaler_params, ddm_pca_model.scaler_aoas, device,
    )

    n = min(n, len(z_inits))
    with torch.no_grad():
        _, _, z_gen = ddm_pca_model.generate(
            z_init=z_inits[:n].to(device),
            params=params_norm[:n].to(device),
            device=device,
        )

    return z_gen.cpu().numpy()   # [n, n_components] — raw (un-normalised) PCA space


def get_train_pca_z(bae_model, pca, train_items, device):
    """Project training BAE latents through the PCA."""
    bae_model.eval()
    z_list = []
    with torch.no_grad():
        for item in train_items:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
            coords_c  = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x  = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)
            x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)
            z_bae  = bae_model.encode(x_wing).cpu().numpy()          # [1, bae_latent_dim]
            z_pca  = pca.transform(z_bae)                             # [1, n_components]
            z_list.append(z_pca[0])

    return np.stack(z_list).astype(np.float32)   # [N_tr, n_components]


# ---------------------------------------------------------------------------
# PCA scatter — PC1 vs PC2, PC1 vs PC3, PC2 vs PC3
# ---------------------------------------------------------------------------

def plot_pca_pairs(train_z, gen_z, pca, save_path, training_only=False):
    _disable_usetex()
    pairs = [(0, 1), (0, 2), (1, 2)]
    ev    = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(train_z[:, i], train_z[:, j],
                   s=14, alpha=0.45, color=C_TRAIN, label="Training set", rasterized=True)
        if not training_only:
            ax.scatter(gen_z[:, i], gen_z[:, j],
                       s=24, alpha=0.70, color=C_GEN, marker="x", label="DDM-PCA",
                       rasterized=True)
        ax.set_xlabel(f"PC{i+1} ({ev[i]:.1%})", fontsize=13)
        ax.set_ylabel(f"PC{j+1} ({ev[j]:.1%})", fontsize=13)
        ax.tick_params(labelsize=11)

    if not training_only:
        axes[0].legend(fontsize=12, markerscale=1.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# LVAE scatter — pair-plot grid for the first N_DIMS dimensions
# ---------------------------------------------------------------------------

def plot_lvae_pairs(train_w, gen_w, n_dims, save_path, dim_labels=None,
                    training_only=False, legend_y=0.07):
    _disable_usetex()
    if dim_labels is None:
        dim_labels = [f"w{i+1}" for i in range(n_dims)]

    fig, axes = plt.subplots(n_dims, n_dims,
                             figsize=(2.3 * n_dims, 2.3 * n_dims),
                             sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.10)

    from matplotlib.lines import Line2D

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]
            if row == col:
                lo = train_w[:, col].min() if training_only else min(train_w[:, col].min(), gen_w[:, col].min())
                hi = train_w[:, col].max() if training_only else max(train_w[:, col].max(), gen_w[:, col].max())
                if hi == lo:
                    hi = lo + 1e-6
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

            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(dim_labels[row], fontsize=24, labelpad=4)
            if row == n_dims - 1:
                ax.set_xlabel(dim_labels[col], fontsize=24, labelpad=4)

    if not training_only:
        handles = [
            Line2D([0], [0], color=C_TRAIN, lw=0, marker="o", markersize=9,
                   alpha=0.7, label="Training set"),
            Line2D([0], [0], color=C_GEN, lw=0, marker="x", markersize=10,
                   label="Generated (DDM-W)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=22,
                   bbox_to_anchor=(0.5, legend_y), frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=None,
                        help="If set, also save a pair plot of only the top-N active dims")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # ── load shared models ────────────────────────────────────────────────
    print("Loading BAE …")
    bae = load_bae_3d(BAE_CKP, DEVICE)

    print("Loading LVAE …")
    lvae = load_lvae_3d(LVAE_CKP, DEVICE, bae)

    print("Loading pre-fitted PCA …")
    with open(PCA_PKL, "rb") as f:
        pca = pickle.load(f)

    # ── dataset ──────────────────────────────────────────────────────────
    dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    train_items = [it for it in dataset["train"] if it["final"] == 1]
    print(f"Training wings: {len(train_items)}")

    # ── LVAE section ─────────────────────────────────────────────────────
    print("Encoding training set through LVAE encoder …")
    train_w = collect_w_latents(lvae, bae, train_items, DEVICE)   # [N_tr, 64]

    print("Loading DDM-W …")
    ddm_w = load_ddm_w(bae, lvae, DEVICE)
    print(f"Generating {N_GEN} wings with DDM-W …")
    gen_w = get_generated_w(ddm_w, bae, lvae, DEVICE, n=N_GEN)

    # Select the active dimensions by variance, not by index
    train_std   = train_w.std(axis=0)                          # [64]
    active_idx  = np.where(train_std >= 0.02)[0]               # indices of active dims
    active_idx  = active_idx[np.argsort(train_std[active_idx])[::-1]]  # sort by std desc
    print(f"Active dimensions ({len(active_idx)}): {active_idx.tolist()}")

    plot_lvae_pairs(
        train_w[:, active_idx], gen_w[:, active_idx],
        n_dims=len(active_idx),
        dim_labels=[f"w{i+1}" for i in active_idx],
        save_path=os.path.join(OUT_DIR, "lvae_latent_dims.pdf"),
    )

    if args.top_n is not None:
        n = min(args.top_n, len(active_idx))
        print(f"Plotting top-{n} pair plot (training + generated) …")
        plot_lvae_pairs(
            train_w[:, active_idx[:n]], gen_w[:, active_idx[:n]],
            n_dims=n,
            dim_labels=[f"w{i+1}" for i in active_idx[:n]],
            legend_y=0.01,
            save_path=os.path.join(OUT_DIR, f"lvae_latent_dims_top{n}.pdf"),
        )
        print(f"Plotting top-{n} pair plot (training only) …")
        plot_lvae_pairs(
            train_w[:, active_idx[:n]], gen_w[:, active_idx[:n]],
            n_dims=n,
            dim_labels=[f"w{i+1}" for i in active_idx[:n]],
            save_path=os.path.join(OUT_DIR, f"lvae_latent_dims_top{n}_train_only.pdf"),
            training_only=True,
        )

    # ── PCA section ──────────────────────────────────────────────────────
    print("Projecting training BAE latents through PCA …")
    train_z = get_train_pca_z(bae, pca, train_items, DEVICE)   # [N_tr, 64]

    print("Loading DDM-PCA …")
    ddm_pca = load_ddm_pca(bae, pca, DEVICE)
    print(f"Generating {N_GEN} PCA z-vectors with DDM-PCA …")
    gen_z = get_generated_pca_z(ddm_pca, bae, pca, DEVICE, n=N_GEN)

    print("Plotting PCA pair-plot (PC1, PC2, PC3) …")
    plot_pca_pairs(
        train_z, gen_z, pca,
        save_path=os.path.join(OUT_DIR, "pca_latent_dims.pdf"),
    )
    print("Plotting PCA pair-plot (training only) …")
    plot_pca_pairs(
        train_z, gen_z, pca,
        save_path=os.path.join(OUT_DIR, "pca_latent_dims_train_only.pdf"),
        training_only=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
