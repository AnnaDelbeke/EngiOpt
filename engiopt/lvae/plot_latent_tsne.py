"""
t-SNE comparison: LVAE latents vs PCA latents vs standard normal prior.

Encodes the test set through the frozen LVAE encoder to get 64-d w-vectors,
transforms the same test coords through a PCA fitted on training data (64 PCs),
draws a matched-size sample from N(0,I), then projects all three to 2-D with
t-SNE and saves a side-by-side scatter comparison.

Usage
-----
    python -m engiopt.lvae.plot_latent_tsne \
        --checkpoint     results/lvae_3d/lvae_3d_v8_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --save_dir       results/lvae_3d_evaluation
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import scienceplots
    plt.style.use("science")
    plt.rcParams["text.usetex"] = False
except ImportError:
    pass

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.train_lvae_3d import (
    LAEEncoder3D, LAEEncoderJoint3D, LAEDecoder3D, LVAE3D,
)
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_coords(item: dict) -> np.ndarray:
    """Return normalised coords as [S, 2, 192] numpy array (same pre-processing as eval)."""
    coords    = torch.tensor(item["coords"],    dtype=torch.float32)   # [S, 192, 2]
    te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)   # [S]
    coords_c  = coords.clone()
    coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
    te_x  = coords_c[:, 0, 0]
    coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
    le_x  = coords_c[:, :, 0].min(dim=1).values
    chord = 1.0 - le_x
    coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)
    return coords_c.permute(0, 2, 1).numpy()   # [S, 2, 192]


def collect_w_latents(model: LVAE3D, bae_model: BezierAutoencoder3D,
                      items: list, device: str,
                      batch_size: int = 32) -> np.ndarray:
    """Encode test items through the LVAE encoder → [N, lae_latent_dim]."""
    model.encoder.eval()
    w_list = []

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        z_baes, pressures, params_scaled = [], [], []

        for item in chunk:
            coords_c  = _normalise_coords(item)                              # [S, 2, 192]
            x_wing    = torch.tensor(coords_c, dtype=torch.float32).unsqueeze(0).to(device)
            # x_wing shape: [1, S, 2, 192] — bae expects [B, S, 2, 192]
            with torch.no_grad():
                z_bae = bae_model.encode(x_wing).squeeze(0).cpu()           # [latent_dim]
            z_baes.append(z_bae)

            pressure = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)
            pressures.append(pressure)

            flow_p = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            params = torch.tensor(flow_p, dtype=torch.float32).unsqueeze(0).to(device)
            params_scaled.append(model.scaler_params.transform(params).cpu())

        z_baes_t   = torch.stack(z_baes).to(device)
        press_t    = torch.stack(pressures).to(device)
        params_t   = torch.cat(params_scaled).to(device)

        with torch.no_grad():
            w = model.encoder(z_baes_t, press_t, params_t)
        w_list.append(w.cpu().numpy())

    return np.concatenate(w_list, axis=0)   # [N, lae_latent_dim]


def fit_pca_latents(train_items: list, test_items: list,
                    n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training coords, return (train_latents, test_latents)."""
    train_np = np.stack([_normalise_coords(it) for it in train_items])   # [N_tr, S, 2, 192]
    test_np  = np.stack([_normalise_coords(it) for it in test_items])    # [N_te, S, 2, 192]
    train_flat = train_np.reshape(len(train_np), -1)
    test_flat  = test_np.reshape(len(test_np),  -1)
    pca = PCA(n_components=n_components, random_state=0)
    train_lat = pca.fit_transform(train_flat).astype(np.float32)
    test_lat  = pca.transform(test_flat).astype(np.float32)
    return train_lat, test_lat


def run_tsne(latents_a: np.ndarray, latents_b: np.ndarray,
             prior: np.ndarray, perplexity: int = 30,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint t-SNE on the concatenation of all three sets; returns split 2-D coords."""
    combined = np.concatenate([latents_a, latents_b, prior], axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, max_iter=1000, init="pca")
    emb  = tsne.fit_transform(combined)
    na, nb = len(latents_a), len(latents_b)
    return emb[:na], emb[na:na + nb], emb[na + nb:]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tsne_comparison(lvae_emb: np.ndarray,
                         pca_emb: np.ndarray,
                         prior_emb: np.ndarray,
                         save_path: str,
                         model_name: str = ""):
    """Three-panel scatter: LVAE | PCA | overlay of both vs prior."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    kw_prior = dict(s=12, alpha=0.35, color="gray",        label="Prior N(0,I)", zorder=1)
    kw_lvae  = dict(s=18, alpha=0.70, color="steelblue",   label="LVAE latents", zorder=2)
    kw_pca   = dict(s=18, alpha=0.70, color="darkorange",  label="PCA latents",  zorder=2)

    # Panel 1: LVAE vs prior
    axes[0].scatter(*prior_emb.T, **kw_prior)
    axes[0].scatter(*lvae_emb.T,  **kw_lvae)
    axes[0].set_title("LVAE latents vs prior")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    # Panel 2: PCA vs prior
    axes[1].scatter(*prior_emb.T, **kw_prior)
    axes[1].scatter(*pca_emb.T,   **kw_pca)
    axes[1].set_title("PCA latents vs prior")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("t-SNE 1")

    # Panel 3: LVAE vs PCA (no prior) — direct structural comparison
    axes[2].scatter(*pca_emb.T,  **kw_pca)
    axes[2].scatter(*lvae_emb.T, **kw_lvae)
    axes[2].set_title("LVAE vs PCA (direct)")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("t-SNE 1")

    suptitle = f"t-SNE latent space comparison"
    if model_name:
        suptitle += f"  [{model_name}]"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to {save_path}")


def plot_tsne_regime(lvae_emb: np.ndarray,
                     pca_emb: np.ndarray,
                     machs: np.ndarray,
                     save_path: str,
                     model_name: str = ""):
    """Two-panel scatter colored by Mach regime: LVAE (left) | PCA (right)."""
    regime_colors = np.where(machs < 0.8, 0, np.where(machs < 1.0, 1, 2)).astype(int)
    palette = ["steelblue", "darkorange", "firebrick"]
    labels  = ["Subsonic (M<0.8)", "Transonic (0.8-1.0)", "Supersonic (M≥1.0)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, title in zip(axes, [lvae_emb, pca_emb], ["LVAE latents", "PCA latents"]):
        for regime_idx in range(3):
            mask = regime_colors == regime_idx
            if mask.any():
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           color=palette[regime_idx], label=labels[regime_idx],
                           s=22, alpha=0.75, edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.legend(fontsize=8, markerscale=1.2)
    axes[0].set_ylabel("t-SNE 2")

    suptitle = "t-SNE colored by Mach regime"
    if model_name:
        suptitle += f"  [{model_name}]"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE regime plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str,
                   default="results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt")
    p.add_argument("--save_dir",        type=str, default="results/lvae_3d_evaluation")
    p.add_argument("--perplexity",      type=int, default=30)
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--n_train_cap",     type=int, default=None,
                   help="Cap training samples used for PCA fit (speeds up for debugging)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bae_model = load_bae_3d(args.bae_checkpoint, device)
    model     = load_lvae_3d(args.checkpoint, device, bae_model)

    dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_items   = [it for it in dataset["test"]  if it["final"] == 1]
    train_items  = [it for it in dataset["train"] if it["final"] == 1]
    if args.n_train_cap:
        train_items = train_items[:args.n_train_cap]

    n_test       = len(test_items)
    n_components = model.lae_latent_dim
    print(f"Test: {n_test}  Train (for PCA): {len(train_items)}  n_components: {n_components}")

    # 1. LVAE w-latents for test set
    print("Encoding test set through LVAE encoder...")
    w_latents = collect_w_latents(model, bae_model, test_items, device)

    # 2. PCA latents
    print("Fitting PCA on training coords and transforming test set...")
    _, pca_test_latents = fit_pca_latents(train_items, test_items, n_components)

    # 3. Prior samples (matched to test set size)
    rng   = np.random.default_rng(args.seed)
    prior = rng.standard_normal((n_test, n_components)).astype(np.float32)

    # 4. Joint t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity}, n={3 * n_test} points)...")
    lvae_emb, pca_emb, prior_emb = run_tsne(
        w_latents, pca_test_latents, prior,
        perplexity=args.perplexity, seed=args.seed,
    )

    # 5. Save plots
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_path  = os.path.join(args.save_dir, f"tsne_latent_{model_name}.png")
    plot_tsne_comparison(lvae_emb, pca_emb, prior_emb,
                         save_path=save_path, model_name=model_name)

    machs = np.array([it["mach"] for it in test_items])
    regime_path = os.path.join(args.save_dir, f"tsne_regime_{model_name}.png")
    plot_tsne_regime(lvae_emb, pca_emb, machs,
                     save_path=regime_path, model_name=model_name)


if __name__ == "__main__":
    main()
