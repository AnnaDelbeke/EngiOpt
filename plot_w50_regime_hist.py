"""
Plot marginal distribution of latent dimension w50 split by flow regime.

Encodes all training wings through the LVAE3D, extracts w50, and produces
three side-by-side histograms (one per regime), colored by the shared
regime palette used throughout the thesis.

Output: results/plots/w50_regime_hist.pdf
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d

_SLICES_PKL = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
_LVAE_CKPT  = "results/lvae_3d/lvae_3d_v29_best.pth"
_BAE_CKPT   = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"

DIM = 49  # w50 label = raw index + 1; raw index 49 is active, raw 50 is pruned

REGIME_COLORS = {
    "subsonic":   "#7BAFD4",
    "transonic":  "#F0A868",
    "supersonic": "#D96B5A",
}
REGIME_LABELS = {
    "subsonic":   "Subsonic\n($M < 0.8$)",
    "transonic":  "Transonic\n($0.8 \leq M < 1.0$)",
    "supersonic": "Supersonic\n($M \geq 1.0$)",
}
REGIME_ORDER = ["subsonic", "transonic", "supersonic"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

bae_model  = load_bae_3d(_BAE_CKPT, device)
lvae_model = load_lvae_3d(_LVAE_CKPT, device, bae_model=bae_model)
lvae_model.encoder.eval()

dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
train_items = [item for item in dataset["train"] if item["final"] == 1]
print(f"Training wings: {len(train_items)}")

ws    = []
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

ws    = np.array(ws)
machs = np.array(machs)
print(f"Encoded {len(ws)} wings, latent dim = {ws.shape[1]}")

regime_masks = {
    "subsonic":   machs < 0.8,
    "transonic":  (machs >= 0.8) & (machs < 1.0),
    "supersonic": machs >= 1.0,
}

all_vals = ws[:, DIM]
global_min, global_max = all_vals.min(), all_vals.max()
bin_width = (global_max - global_min) / 30
shared_bins = np.arange(global_min, global_max + bin_width, bin_width)

import os
os.makedirs("results/plots", exist_ok=True)

for reg in REGIME_ORDER:
    mask  = regime_masks[reg]
    vals  = ws[mask, DIM]
    color = REGIME_COLORS[reg]

    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.hist(vals, bins=shared_bins, color=color, alpha=0.85, edgecolor="none")
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = f"results/plots/w50_regime_hist_{reg}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

print(f"Saved {out}")
