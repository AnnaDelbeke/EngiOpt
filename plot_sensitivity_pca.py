"""
PCA of the 100 sensitivity-experiment outputs to probe uni- vs multimodality.

For each anchor, 10 wings were generated with identical flow conditions but
different w_init. This script flattens each generated wing's coordinates
(15 slices x 2D coords x 192 points -> 1D vector), runs PCA across all 100
outputs, and plots the first two PCs coloured by anchor.

If the model is unimodal: 10 points per anchor cluster tightly around a single
centroid, with no within-anchor sub-structure.
If multimodal: within-anchor points spread into distinct sub-clusters.
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import numpy as np

JSON_PATH = "results/sensitivity_3d/sensitivity_ddm_w_20260610_094143.json"
OUT_PATH  = "thesis/figures/sensitivity_pca.pdf"

REGIME_COLORS = {
    "subsonic":   "#4477AA",
    "transonic":  "#EE7733",
    "supersonic": "#CC3311",
}
REGIME_MARKERS = {
    "subsonic":   "o",
    "transonic":  "s",
    "supersonic": "^",
}

def main():
    with open(JSON_PATH) as f:
        d = json.load(f)

    anchors = d["per_anchor"]

    # --- flatten all 100 generated wings into vectors -----------------------
    # coords_list[i] has shape [15, 2, 192] stored as nested lists
    vecs, labels, regimes, gt_vecs = [], [], [], []
    for anchor_idx, anchor in enumerate(anchors):
        for coords in anchor["coords_list"]:
            flat = np.array(coords).ravel()
            vecs.append(flat)
            labels.append(anchor_idx)
            regimes.append(anchor["regime"])
        gt_flat = np.array(anchor["gt_coords"]).ravel()
        gt_vecs.append(gt_flat)

    vecs    = np.stack(vecs)     # [100, D]
    gt_vecs = np.stack(gt_vecs) # [10, D]

    # --- PCA on generated wings only ----------------------------------------
    pca = PCA(n_components=2, random_state=0)
    pca.fit(vecs)
    coords_2d = pca.transform(vecs)    # [100, 2]
    gt_2d     = pca.transform(gt_vecs) # [10, 2]

    var = pca.explained_variance_ratio_

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # one scatter per anchor so we can draw convex hulls / ellipses later
    for anchor_idx, anchor in enumerate(anchors):
        mask   = [i for i, l in enumerate(labels) if l == anchor_idx]
        pts    = coords_2d[mask]
        regime = anchor["regime"]
        color  = REGIME_COLORS[regime]
        marker = REGIME_MARKERS[regime]

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, marker=marker, s=60, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)

        # centroid
        ctr = pts.mean(axis=0)
        ax.scatter(*ctr, c=color, marker=marker, s=180,
                   edgecolors="black", linewidths=1.2, zorder=5)

        # label anchor with case number
        ax.annotate(f"#{anchor['case_num']}\nM={anchor['mach']:.2f}",
                    xy=ctr, xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color=color)

    # ground-truth optimised wings (stars)
    for anchor_idx, anchor in enumerate(anchors):
        regime = anchor["regime"]
        ax.scatter(*gt_2d[anchor_idx], marker="*", s=200,
                   c=REGIME_COLORS[regime],
                   edgecolors="black", linewidths=0.8, zorder=6)

    # legend
    patches = [mpatches.Patch(color=c, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    gen_h  = ax.scatter([], [], c="gray", s=60, marker="o", label="Generated (small)")
    ctr_h  = ax.scatter([], [], c="gray", s=180, marker="o",
                        edgecolors="black", label="Anchor centroid (large)")
    gt_h   = ax.scatter([], [], c="gray", s=200, marker="*",
                        edgecolors="black", label="Ground truth (star)")
    ax.legend(handles=patches + [gen_h, ctr_h, gt_h],
              fontsize=8, loc="best", framealpha=0.9)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA of 100 sensitivity outputs\n"
                 "Small markers = individual inits, large = anchor centroid, "
                 "star = ground truth", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")

    # --- text summary -------------------------------------------------------
    print("\nWithin-anchor spread (std of PC1, PC2 per anchor):")
    for anchor_idx, anchor in enumerate(anchors):
        mask = [i for i, l in enumerate(labels) if l == anchor_idx]
        pts  = coords_2d[mask]
        print(f"  #{anchor['case_num']} M={anchor['mach']:.2f} "
              f"std_PC1={pts[:,0].std():.3f}  std_PC2={pts[:,1].std():.3f}")

    print("\nBetween-anchor spread (std of centroids):")
    centroids = []
    for anchor_idx in range(len(anchors)):
        mask = [i for i, l in enumerate(labels) if l == anchor_idx]
        centroids.append(coords_2d[mask].mean(axis=0))
    centroids = np.stack(centroids)
    print(f"  std_PC1={centroids[:,0].std():.3f}  std_PC2={centroids[:,1].std():.3f}")
    print("\nRatio between/within (>1 means anchors are well separated):")
    within = np.mean([coords_2d[[i for i,l in enumerate(labels) if l==a]][:,0].std()
                      for a in range(len(anchors))])
    between = centroids[:,0].std()
    print(f"  PC1: {between/within:.2f}x")


if __name__ == "__main__":
    main()
