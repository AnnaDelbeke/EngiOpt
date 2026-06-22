"""Regenerate the latent std figure for the thesis using the v29 LVAE checkpoint."""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, ".")
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d, encode_item_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"
TAU      = 0.02

device = "cpu"
bae  = load_bae_3d(BAE_CKP, device)
lvae = load_lvae_3d(LVAE_CKP, device, bae)

ds    = NewWingsDataset("Wing_TL/data/processed/new_dataset_slices.pkl",
                        "Wing_TL/data/processed/new_dataset_scalars.pkl", seed=0)
items = [it for it in list(ds["train"]) if it["final"] == 1]
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
print(f"Stds range: {stds.min():.6f} – {stds.max():.4f}")

mask  = lvae.active_latent_mask.cpu().numpy()
order = np.argsort(stds)[::-1]
stds_s = stds[order]
mask_s = mask[order]

n_active = int(mask_s.sum())
n_pruned = int((~mask_s).sum())

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

for path in (
    "results/lvae_3d_evaluation/latent_std_lvae_3d_v29_best",
    "thesis/figures/latent_std_lvae_3d_v29_best",
):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {path}.{ext}")

plt.close()
