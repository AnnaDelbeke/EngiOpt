"""
Hartigan dip test on LVAE latent codes of 767 training wings.

Encodes each training wing through the LVAE3D to get its w vector,
groups wings by Mach regime, and tests each latent dimension for
unimodality using the Hartigan dip statistic.

Usage:
    .venv/bin/python run_dip_test_training_wings.py
"""

import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
_LVAE_CKPT   = "results/lvae_3d/lvae_3d_v29_best.pth"
_BAE_CKPT    = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load models ──────────────────────────────────────────────────────────────
bae_model = load_bae_3d(_BAE_CKPT, device)
lvae_model = load_lvae_3d(_LVAE_CKPT, device, bae_model=bae_model)
lvae_model.encoder.eval()

# ── Load training wings ───────────────────────────────────────────────────────
print("Loading dataset...")
dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
train_items = [item for item in dataset["train"] if item["final"] == 1]
print(f"Training wings: {len(train_items)}")

# ── Encode all wings through LVAE ────────────────────────────────────────────
print("Encoding wings...")
ws = []
machs = []
with torch.no_grad():
    for i, item in enumerate(train_items):
        if i % 100 == 0:
            print(f"  {i}/{len(train_items)}")
        z_bae, gt_recon, aoa, params_scaled, te_shifts, pressure, le_x, chord = \
            encode_item_3d(item, bae_model, lvae_model, device)

        z_bae_b    = z_bae.unsqueeze(0).to(device)
        pressure_b = pressure.unsqueeze(0).to(device)
        params_b   = params_scaled.to(device)

        w = lvae_model.encoder(z_bae_b, pressure_b, params_b)
        w = lvae_model._apply_mask(w)
        ws.append(w.squeeze(0).cpu().numpy())
        machs.append(item["mach"])

ws    = np.array(ws)     # (N, latent_dim)
machs = np.array(machs)

print(f"Encoded {len(ws)} wings, latent dim = {ws.shape[1]}")

# ── Hartigan dip test ─────────────────────────────────────────────────────────
try:
    from diptest import diptest
    def dip_pvalue(x):
        _, p = diptest(x)
        return p
except ImportError:
    # Fallback: simple dip statistic via scipy / manual
    print("diptest not installed, trying diptest via pip...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "diptest", "-q"])
    from diptest import diptest
    def dip_pvalue(x):
        _, p = diptest(x)
        return p

def regime(m):
    if m < 0.8:   return "subsonic"
    if m < 1.0:   return "transonic"
    return "supersonic"

regime_masks = {
    "subsonic":    machs < 0.8,
    "transonic":   (machs >= 0.8) & (machs < 1.0),
    "supersonic":  machs >= 1.0,
}

# ── Get active dimensions from mask ──────────────────────────────────────────
active_dims = None
if hasattr(lvae_model, "active_latent_mask"):
    mask = lvae_model.active_latent_mask.cpu().numpy().astype(bool)
    active_dims = np.where(mask)[0]
    print(f"Active latent dimensions: {len(active_dims)} / {len(mask)}")
else:
    active_dims = np.arange(ws.shape[1])

print("\n" + "="*60)
print("HARTIGAN DIP TEST — LVAE latent space (training wings)")
print("="*60)
print(f"{'Regime':<15} {'n':>5}  {'Dims sig. (p<0.05)':>20}  {'Min p-val dim'}")
print("-"*60)

ALPHA = 0.05

all_results = {}
for reg, mask in regime_masks.items():
    w_reg = ws[mask][:, active_dims]
    n     = len(w_reg)
    pvals = np.array([dip_pvalue(w_reg[:, d]) for d in range(w_reg.shape[1])])
    n_sig = (pvals < ALPHA).sum()
    min_p = pvals.min()
    min_d = active_dims[pvals.argmin()]
    all_results[reg] = {"n": n, "pvals": pvals, "n_sig": n_sig, "min_p": min_p, "min_d": min_d}
    print(f"{reg:<15} {n:>5}  {n_sig:>20}  dim {min_d} (p={min_p:.4f})")

# ── Overall (all regimes pooled) ──────────────────────────────────────────────
w_all  = ws[:, active_dims]
pvals_all = np.array([dip_pvalue(w_all[:, d]) for d in range(w_all.shape[1])])
n_sig_all = (pvals_all < ALPHA).sum()
min_p_all = pvals_all.min()
min_d_all = active_dims[pvals_all.argmin()]
print(f"{'all (pooled)':<15} {len(ws):>5}  {n_sig_all:>20}  dim {min_d_all} (p={min_p_all:.4f})")

print("\n" + "="*60)
print("Per-regime detail (p < 0.05 means evidence of multimodality)")
print("="*60)
for reg, res in all_results.items():
    sig_dims = active_dims[res["pvals"] < ALPHA]
    print(f"\n{reg.upper()} (n={res['n']})")
    if len(sig_dims) == 0:
        print(f"  All {len(res['pvals'])} active dims unimodal (all p ≥ {ALPHA})")
        print(f"  Smallest p-value: {res['min_p']:.4f} (dim {res['min_d']})")
    else:
        print(f"  {len(sig_dims)} dims with p < {ALPHA}: {sig_dims.tolist()}")
        for d_idx in sig_dims:
            local_i = np.where(active_dims == d_idx)[0][0]
            print(f"    dim {d_idx}: p = {res['pvals'][local_i]:.4f}")

print("\nDone.")
