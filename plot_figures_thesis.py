"""
Generate two thesis figures:
  1. sensitivity_ddmw_bars.pdf  — sensitivity bars sorted by Mach
  2. ablation_ddmw_v29.pdf      — DDM-W v29 ablation dual-axis plot
"""
import json, glob, re, os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)


# ─── 1. Sensitivity bars sorted by Mach ────────────────────────────────────

JSON = "results/sensitivity_3d/sensitivity_ddm_w_20260618_122438.json"
with open(JSON) as f:
    data = json.load(f)

results = sorted(data["per_anchor"], key=lambda r: r["mach"])

labels   = [f"M={r['mach']:.3f}" for r in results]
mses     = [r["pairwise_shape_mse"] for r in results]
gt_mses  = [r["mean_gt_mse"] for r in results]
REGIME_COLORS = {"subsonic": "#7BAFD4", "transonic": "#F0A868", "supersonic": "#D96B5A"}
colors   = [
    REGIME_COLORS["subsonic"] if r["mach"] < 0.8 else (REGIME_COLORS["transonic"] if r["mach"] < 1.0 else REGIME_COLORS["supersonic"])
    for r in results
]

legend_elements = [
    Patch(facecolor=REGIME_COLORS["subsonic"],   label="Subsonic ($M<0.8$)"),
    Patch(facecolor=REGIME_COLORS["transonic"],  label="Transonic ($0.8$–$1.0$)"),
    Patch(facecolor=REGIME_COLORS["supersonic"], label="Supersonic ($M\geq 1.0$)"),
]

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.bar(range(len(labels)), mses, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel(r"Mean pairwise Shape MSE  $S_\mathrm{init}$", fontsize=10)
ax.legend(handles=legend_elements, loc="upper left", ncol=1, fontsize=8, frameon=False)
fig.tight_layout()
out1a = f"{OUT}/sensitivity_ddmw_bars_a.pdf"
fig.savefig(out1a, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1a}")

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.bar(range(len(labels)), gt_mses, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Mean Shape MSE vs ground truth", fontsize=10)
fig.tight_layout()
out1b = f"{OUT}/sensitivity_ddmw_bars_b.pdf"
fig.savefig(out1b, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1b}")


# ─── 2. Regime strip plot ──────────────────────────────────────────────────

regime_order  = ["subsonic", "transonic", "supersonic"]
regime_labels = {"subsonic": "Subsonic\n($M < 0.8$)", "transonic": "Transonic\n($0.8$–$1.0$)", "supersonic": "Supersonic\n($M \geq 1.0$)"}

by_regime = {r: [] for r in regime_order}
for r in data["per_anchor"]:
    by_regime[r["regime"]].append(r["pairwise_shape_mse"])

fig, ax = plt.subplots(figsize=(5, 4))
for i, regime in enumerate(regime_order):
    vals = by_regime[regime]
    color = REGIME_COLORS[regime]
    ax.scatter([i] * len(vals), vals, color=color, s=60, zorder=3)
    ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=color, linewidth=2.0, zorder=4)

ax.set_xticks(range(len(regime_order)))
ax.set_xticklabels([regime_labels[r] for r in regime_order], fontsize=9)
ax.set_ylabel(r"Pairwise Shape MSE $S_\mathrm{init}$", fontsize=10)
fig.tight_layout()
out1c = f"{OUT}/sensitivity_ddmw_strip.pdf"
fig.savefig(out1c, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1c}")


# ─── 3. DDM-W v29 ablation dual-axis plot ──────────────────────────────────

base = "results/ddm_w_3d_ablation_v29"
rows = []
for jf in glob.glob(f"{base}/*/eval_*_best_*.json"):
    m = re.search(r"_n(\d+)_s(\d+)_", jf)
    if not m:
        continue
    n, s = int(m.group(1)), int(m.group(2))
    with open(jf) as f:
        d = json.load(f)
    rows.append((n, s, d["shape_mse"], d["vendi"]))

rows.sort()
ns = sorted(set(r[0] for r in rows))

# collect per-n: list of (mse, vendi) across seeds
from collections import defaultdict
by_n = defaultdict(lambda: {"mse": [], "vendi": []})
for n, s, mse, vendi in rows:
    by_n[n]["mse"].append(mse)
    by_n[n]["vendi"].append(vendi)

ns_arr      = np.array(ns)
mse_mean    = np.array([np.mean(by_n[n]["mse"])   for n in ns])
mse_std     = np.array([np.std(by_n[n]["mse"])    for n in ns])
vendi_mean  = np.array([np.mean(by_n[n]["vendi"]) for n in ns])
vendi_std   = np.array([np.std(by_n[n]["vendi"])  for n in ns])

fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

color_mse   = "#2166ac"
color_vendi = "#d6604d"

# plot each seed as a thin line, mean as thick
seed0_mse   = [by_n[n]["mse"][0]   for n in ns]
seed1_mse   = [by_n[n]["mse"][1]   for n in ns]
seed0_vendi = [by_n[n]["vendi"][0] for n in ns]
seed1_vendi = [by_n[n]["vendi"][1] for n in ns]

ax1.plot(ns_arr, seed0_mse,   color=color_mse,   linewidth=0.8, linestyle="-",  alpha=0.4, marker="o", markersize=3)
ax1.plot(ns_arr, seed1_mse,   color=color_mse,   linewidth=0.8, linestyle="--", alpha=0.4, marker="o", markersize=3)
l1, = ax1.plot(ns_arr, mse_mean, color=color_mse, linewidth=2.2, marker="o", markersize=6, label="Shape MSE (mean of 2 seeds)")

ax2.plot(ns_arr, seed0_vendi, color=color_vendi, linewidth=0.8, linestyle="-",  alpha=0.4, marker="s", markersize=3)
ax2.plot(ns_arr, seed1_vendi, color=color_vendi, linewidth=0.8, linestyle="--", alpha=0.4, marker="s", markersize=3)
l2, = ax2.plot(ns_arr, vendi_mean, color=color_vendi, linewidth=2.2, marker="s", markersize=6, label="Vendi (mean of 2 seeds)", linestyle="--")

# phase-transition annotation
ax1.axvspan(150, 300, color="gold", alpha=0.18, label="Phase transition")

ax1.set_xlabel("Training set size $n$", fontsize=10)
ax1.set_ylabel("Shape MSE", fontsize=10, color=color_mse)
ax2.set_ylabel("Normalised Vendi score", fontsize=10, color=color_vendi)
ax1.tick_params(axis="y", labelcolor=color_mse)
ax2.tick_params(axis="y", labelcolor=color_vendi)
ax1.set_xticks(ns_arr)
ax1.set_xticklabels([str(n) for n in ns], rotation=45, ha="right", fontsize=8)

lines = [l1, l2, plt.Rectangle((0,0),1,1, color="gold", alpha=0.4)]
labels_leg = ["Shape MSE (mean of 2 seeds)", "Vendi score (mean of 2 seeds)", "Phase transition"]
ax1.legend(lines, labels_leg, fontsize=8, frameon=False, loc="upper right")

fig.tight_layout()
out2 = f"{OUT}/ablation_ddmw_v29.pdf"
fig.savefig(out2, dpi=150, bbox_inches="tight")
fig.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ─── 3. Wing overlays per regime ───────────────────────────────────────────

SPAN_LABELS = {0: r"$z = 0.00$", 4: r"$z = 0.64$",
               8: r"$z = 1.29$", 14: r"$z = 2.25$"}
SLICE_INDICES = [0, 4, 8, 14]
REGIME_ORDER  = ["subsonic", "transonic", "supersonic"]
REGIME_PLOT_LABELS = {
    "subsonic":   "Subsonic\n($M < 0.8$)",
    "transonic":  "Transonic\n($0.8 \leq M < 1.0$)",
    "supersonic": "Supersonic\n($M \geq 1.0$)",
}

with open(JSON) as f:
    sens_data = json.load(f)

# Pick one representative anchor per regime (highest pairwise MSE = most interesting)
rep = {}
for r in sens_data["per_anchor"]:
    regime = r["regime"]
    if regime not in rep or r["pairwise_shape_mse"] > rep[regime]["pairwise_shape_mse"]:
        rep[regime] = r

for regime in REGIME_ORDER:
    r = rep[regime]
    coords_list = np.array(r["coords_list"])  # (n_inits, 15, 2, 192)
    gt_coords   = np.array(r["gt_coords"])    # (15, 2, 192)
    color = REGIME_COLORS[regime]

    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))

    for col, s_idx in enumerate(SLICE_INDICES):
        ax = axes[col]
        for init_coords in coords_list:
            sl = init_coords[s_idx]
            ax.plot(sl[0], sl[1], color=color, alpha=0.35, linewidth=0.8)
        gt_sl = gt_coords[s_idx]
        ax.plot(gt_sl[0], gt_sl[1], color="black", linewidth=1.4,
                linestyle="--", zorder=5, label="GT" if col == 0 else None)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(SPAN_LABELS[s_idx], fontsize=13)
        if col == 0:
            ax.legend(fontsize=7, frameon=False,
                      bbox_to_anchor=(0.5, -0.15), loc="upper center")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.05)
    out = f"{OUT}/sensitivity_ddmw_wings_{regime}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─── 5. Paired init → output plot ─────────────────────────────────────────
# For one anchor per regime (highest S_init), show 3 paired rows:
#   left col  = init wing cross-section at mid-span
#   right col = generated output cross-section at mid-span

N_PAIRS   = 3
MID_SLICE = 8  # z = 1.29 m, mid-span

for regime in REGIME_ORDER:
    r            = rep[regime]
    coords_list      = np.array(r["coords_list"])       # (30, 15, 2, 192)
    init_coords_list = np.array(r["init_coords_list"])  # (30, 15, 2, 192)
    gt_coords        = np.array(r["gt_coords"])          # (15, 2, 192)
    color            = REGIME_COLORS[regime]

    # pick 3 inits spread across the range of output diversity
    pairwise_dists = [
        float(((coords_list[i] - gt_coords) ** 2).mean())
        for i in range(len(coords_list))
    ]
    sorted_idx = np.argsort(pairwise_dists)
    # pick low, mid, high GT-distance to show range
    chosen = [
        sorted_idx[0],
        sorted_idx[len(sorted_idx) // 2],
        sorted_idx[-1],
    ]

    fig, axes = plt.subplots(N_PAIRS, 2, figsize=(5, N_PAIRS * 1.4))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row, idx in enumerate(chosen):
        # left: init wing
        ax_init = axes[row, 0]
        sl_init = init_coords_list[idx][MID_SLICE]
        ax_init.plot(sl_init[0], sl_init[1], color="#888888", linewidth=1.2)
        ax_init.set_aspect("equal")
        ax_init.axis("off")
        if row == 0:
            ax_init.set_title("Starting wing\n($\\mathbf{w}_\\mathrm{init}$)", fontsize=11)

        # right: generated output
        ax_out = axes[row, 1]
        sl_out = coords_list[idx][MID_SLICE]
        ax_out.plot(sl_out[0], sl_out[1], color=color, linewidth=1.2)
        gt_sl = gt_coords[MID_SLICE]
        ax_out.plot(gt_sl[0], gt_sl[1], color="black", linewidth=1.0,
                    linestyle="--", zorder=5)
        ax_out.set_aspect("equal")
        ax_out.axis("off")
        if row == 0:
            ax_out.set_title("Generated output", fontsize=11)

        # arrow between columns
        fig.text(0.5, axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2,
                 r"$\rightarrow$", ha="center", va="center", fontsize=14)

    # legend for GT dashed line
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="GT")]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=11,
               frameon=False, ncol=1, bbox_to_anchor=(0.75, 0.01))

    out = f"{OUT}/sensitivity_ddmw_paired_{regime}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
