"""Plot BAE-3D data ablation: test MSE vs n_training_samples + midspan recon strip."""

import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

N_SAMPLES = [50, 100, 150, 200, 300, 400, 500, 600, 700, 767]
SEEDS     = [0, 1]

BASE      = "results/bezier_ae_3d_ablation"
FULL_CKPT = "results/bezier_ae_3d/run_039"

# ── collect MSE results ──────────────────────────────────────────────────────
results = {}
for n in N_SAMPLES:
    vals = []
    if n == 767:
        path = f"{BASE}/n767_s0/eval_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                vals.append(json.load(f)["mse_mean"])
    else:
        for s in SEEDS:
            path = f"{BASE}/n{n}_s{s}/eval_metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    vals.append(json.load(f)["mse_mean"])
    if vals:
        results[n] = vals

ns    = sorted(results.keys())
means = [np.mean(results[n]) for n in ns]
stds  = [np.std(results[n])  for n in ns]

# ── load test dataset and checkpoints for airfoil panels ─────────────────────
SHOW_NS  = [50, 200, 400, 767]
WING_IDX = 8
MIDSPAN  = 7

ds = NewWingsDataset("Wing_TL/data/processed/new_dataset_slices.pkl",
                     "Wing_TL/data/processed/new_dataset_scalars.pkl", seed=0)
test_items = [it for it in list(ds["test"]) if it["final"] == 1]
test_ds = WingsBezierDataset3D(test_items)
x_gt = test_ds[WING_IDX][MIDSPAN]  # [2, 192]

def load_model(n):
    if n == 767:
        ckpt_path = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
    else:
        ckpt_path = f"{BASE}/n{n}_s0/models/bae_3d_ablation_n{n}_s0_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BezierAutoencoder3D(n_spans=15, n_control_points=32, n_data_points=192,
        slice_hidden_dims=ckpt.get("slice_hidden_dims", [256,256,256]),
        span_hidden_dims=ckpt.get("span_hidden_dims", [256,256]),
        latent_dim=ckpt.get("latent_dim", 128))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

show_ns = [n for n in SHOW_NS if n in results]

os.makedirs("results/bezier_ae_3d_ablation", exist_ok=True)
os.makedirs("thesis/figures", exist_ok=True)

# ── (a) MSE curve ─────────────────────────────────────────────────────────────
fig_a, ax = plt.subplots(figsize=(5, 4))
ax.plot(ns, means, color="#4477AA", marker="o", linewidth=1.5, markersize=5)
ax.fill_between(ns,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color="#4477AA", alpha=0.2)
ax.set_ylabel("Test MSE", fontsize=10)
ax.set_xlabel("Number of training wings", fontsize=10)
ax.set_xticks(ns)
ax.tick_params(axis="x", rotation=45, labelsize=8)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.tick_params(axis="y", direction="out", which="both", right=False)
ax.tick_params(axis="x", which="both", direction="out", top=False)
fig_a.tight_layout()
for path in ("results/bezier_ae_3d_ablation/ablation_bae3d_curve",
             "thesis/figures/ablation_bae3d_curve"):
    for ext in ("pdf", "png"):
        fig_a.savefig(f"{path}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {path}.{ext}")
plt.close(fig_a)

# ── (b) 2×2 airfoil grid drawn directly ──────────────────────────────────────
fig_b, axes_b = plt.subplots(2, 2, figsize=(5, 2.2),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.05})
x_np = x_gt.numpy()
for idx, n in enumerate(show_ns[:4]):
    r, c = divmod(idx, 2)
    ax_b = axes_b[r, c]
    model = load_model(n)
    x_in = test_ds[WING_IDX].unsqueeze(0)
    with torch.no_grad():
        y, _ = model(x_in)
    y_np = y.squeeze(0)[MIDSPAN].numpy()
    ax_b.plot(x_np[0], x_np[1], color="#4477AA", lw=1.5)
    ax_b.plot(y_np[0], y_np[1], color="#EE6677", lw=1.5, linestyle="--")
    ax_b.set_xlim(-0.05, 1.05)
    ax_b.set_ylim(-0.10, 0.16)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.set_title(f"n = {n}", fontsize=9, pad=3)
for path in ("results/bezier_ae_3d_ablation/ablation_bae3d_airfoils",
             "thesis/figures/ablation_bae3d_airfoils"):
    for ext in ("pdf", "png"):
        fig_b.savefig(f"{path}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {path}.{ext}")
plt.close(fig_b)

# ── combined (for quick preview) ──────────────────────────────────────────────
for path in ("results/bezier_ae_3d_ablation/ablation_bae3d_metrics",
             "thesis/figures/ablation_bae3d_metrics"):
    import shutil
    shutil.copy(f"thesis/figures/ablation_bae3d_curve.png", f"{path}.png") if path.startswith("results") else None
print("Done.")
