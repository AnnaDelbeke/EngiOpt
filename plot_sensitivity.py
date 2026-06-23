"""
Sensitivity analysis plots.

Usage:
    .venv/bin/python plot_sensitivity.py [--json results/sensitivity_3d/sensitivity_ddm_w_*.json]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

OUT = "thesis/figures"
os.makedirs(OUT, exist_ok=True)

REGIME_COLORS = {
    "subsonic":   "#7BAFD4",
    "transonic":  "#F0A868",
    "supersonic": "#D96B5A",
}
REGIME_LABELS = {
    "subsonic":   "Subsonic\n($M < 0.8$)",
    "transonic":  "Transonic\n($0.8$--$1.0$)",
    "supersonic": "Supersonic\n($M \geq 1.0$)",
}
REGIME_ORDER = ["subsonic", "transonic", "supersonic"]
SLICE_INDICES = [0, 4, 8, 14]
SPAN_LABELS = {0: r"$z = 0.00$", 4: r"$z = 0.64$",
               8: r"$z = 1.29$", 14: r"$z = 2.25$"}


def plot_bars(data, out_stem):
    results = sorted(data["per_anchor"], key=lambda r: r["mach"])
    labels  = [f"M={r['mach']:.3f}" for r in results]
    colors  = [
        REGIME_COLORS["subsonic"] if r["mach"] < 0.8
        else (REGIME_COLORS["transonic"] if r["mach"] < 1.0
              else REGIME_COLORS["supersonic"])
        for r in results
    ]
    legend_elements = [
        Patch(facecolor=REGIME_COLORS["subsonic"],   label="Subsonic ($M<0.8$)"),
        Patch(facecolor=REGIME_COLORS["transonic"],  label="Transonic ($0.8$--$1.0$)"),
        Patch(facecolor=REGIME_COLORS["supersonic"], label="Supersonic ($M\geq 1.0$)"),
    ]

    # Pairwise shape MSE bars
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(range(len(labels)), [r["pairwise_shape_mse"] for r in results],
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Mean pairwise Shape MSE  $S_\mathrm{init}$", fontsize=10)
    ax.legend(handles=legend_elements, loc="upper left", ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_stem}_bars.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_stem}_bars.pdf")

    # GT MSE bars
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(range(len(labels)), [r["mean_gt_mse"] for r in results],
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Shape MSE vs ground truth", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_stem}_gt_bars.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_stem}_gt_bars.pdf")

    # Regime strip plot
    by_regime = {r: [] for r in REGIME_ORDER}
    for r in data["per_anchor"]:
        by_regime[r["regime"]].append(r["pairwise_shape_mse"])

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, regime in enumerate(REGIME_ORDER):
        vals  = by_regime[regime]
        color = REGIME_COLORS[regime]
        ax.scatter([i] * len(vals), vals, color=color, s=60, zorder=3)
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=color, linewidth=2.0, zorder=4)
    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=9)
    ax.set_ylabel(r"Pairwise Shape MSE $S_\mathrm{init}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_stem}_strip.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_stem}_strip.pdf")


def plot_wing_overlays(data, out_stem):
    rep = {}
    for r in data["per_anchor"]:
        regime = r["regime"]
        if regime not in rep or r["pairwise_shape_mse"] > rep[regime]["pairwise_shape_mse"]:
            rep[regime] = r

    for regime in REGIME_ORDER:
        if regime not in rep:
            continue
        r          = rep[regime]
        coords_list = np.array(r["coords_list"])  # (n_inits, 15, 2, 192)
        gt_coords   = np.array(r["gt_coords"])    # (15, 2, 192)
        color       = REGIME_COLORS[regime]

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
            ax.set_title(SPAN_LABELS.get(s_idx, str(s_idx)), fontsize=13)
            if col == 0:
                ax.legend(fontsize=7, frameon=False,
                          bbox_to_anchor=(0.5, -0.15), loc="upper center")

        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.05)
        out = f"{out_stem}_wings_{regime}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def plot_paired(data, out_stem):
    rep = {}
    for r in data["per_anchor"]:
        regime = r["regime"]
        if regime not in rep or r["pairwise_shape_mse"] > rep[regime]["pairwise_shape_mse"]:
            rep[regime] = r

    N_PAIRS   = 3
    MID_SLICE = 8

    for regime in REGIME_ORDER:
        if regime not in rep:
            continue
        r                = rep[regime]
        coords_list      = np.array(r["coords_list"])
        init_coords_list = np.array(r["init_coords_list"])
        gt_coords        = np.array(r["gt_coords"])
        color            = REGIME_COLORS[regime]

        pairwise_dists = [float(((coords_list[i] - gt_coords) ** 2).mean())
                          for i in range(len(coords_list))]
        sorted_idx = np.argsort(pairwise_dists)
        chosen = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]

        fig, axes = plt.subplots(N_PAIRS, 2, figsize=(5, N_PAIRS * 1.4))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for row, idx in enumerate(chosen):
            ax_init = axes[row, 0]
            sl_init = init_coords_list[idx][MID_SLICE]
            ax_init.plot(sl_init[0], sl_init[1], color="#888888", linewidth=1.2)
            ax_init.set_aspect("equal")
            ax_init.axis("off")
            if row == 0:
                ax_init.set_title("Starting wing\n($\\mathbf{w}_\\mathrm{init}$)", fontsize=11)

            ax_out = axes[row, 1]
            sl_out = coords_list[idx][MID_SLICE]
            ax_out.plot(sl_out[0], sl_out[1], color=color, linewidth=1.2)
            gt_sl  = gt_coords[MID_SLICE]
            ax_out.plot(gt_sl[0], gt_sl[1], color="black", linewidth=1.0,
                        linestyle="--", zorder=5)
            ax_out.set_aspect("equal")
            ax_out.axis("off")
            if row == 0:
                ax_out.set_title("Generated output", fontsize=11)

            fig.text(0.5,
                     axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2,
                     r"$\rightarrow$", ha="center", va="center", fontsize=14)

        legend_elements = [Line2D([0], [0], color="black", linestyle="--",
                                   linewidth=1.0, label="GT")]
        fig.legend(handles=legend_elements, loc="lower center", fontsize=11,
                   frameon=False, ncol=1, bbox_to_anchor=(0.75, 0.01))

        out = f"{out_stem}_paired_{regime}.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=None,
                   help="Sensitivity JSON file(s). Defaults to most recent in results/sensitivity_3d/")
    args = p.parse_args()

    if args.json:
        json_files = args.json
    else:
        json_files = sorted(glob.glob("results/sensitivity_3d/sensitivity_ddm_w_*.json"))
        if not json_files:
            json_files = sorted(glob.glob("results/sensitivity_3d/sensitivity_*.json"))

    for jf in json_files:
        if not os.path.exists(jf):
            print(f"Missing: {jf}")
            continue
        with open(jf) as f:
            data = json.load(f)
        stem = os.path.join(OUT, os.path.splitext(os.path.basename(jf))[0])
        print(f"\nProcessing {jf}  ({len(data['per_anchor'])} anchors)")
        plot_bars(data, stem)
        if "coords_list" in data["per_anchor"][0]:
            plot_wing_overlays(data, stem)
            plot_paired(data, stem)


if __name__ == "__main__":
    main()
