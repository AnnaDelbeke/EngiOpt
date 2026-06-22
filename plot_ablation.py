"""
Reads eval JSON files from results/ddm_w_3d_ablation_v29/ and produces:
  1. Metrics curves: one panel per metric, x = training data size,
     y = mean across seeds, shaded band = std across seeds.
  2. Wing composite: one column per training size, showing the wing3d_s0
     image for each size so the visual quality progression is clear.

Usage:
    python plot_ablation.py [--ablation_dir results/ddm_w_3d_ablation_v29]
                            [--out results/ablation_v29_plot.png]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


METRICS = {
    "shape_mse":             "Shape MSE",
    "aoa_mse":               "AoA MSE",
    "pressure_mse":          "Pressure MSE",
    "volume_mse":            "Volume MSE",
    "mmd":                   "MMD (Shape space)",
    "mmd_w":                 "Latent Coverage MMD",
    "mmd_z":                 "MMD (Latent z)",
    "vendi":                 "Vendi Score",
    "volume_constraint_sat": "Volume Constraint Sat.",
}

THESIS_METRICS = ["mmd", "aoa_mse", "vendi", "shape_mse"]


def find_eval_jsons(ablation_dir):
    """Return dict: n_samples -> list of metric dicts, one per seed."""
    by_n = defaultdict(list)
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s*"))):
        # parse n and s from directory name
        m = re.search(r"n(\d+)_s(\d+)", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        jsons = sorted(glob.glob(os.path.join(n_dir, "eval_*.json")))
        if not jsons:
            continue
        # take the most recent JSON if multiple exist
        with open(jsons[-1]) as f:
            data = json.load(f)
        by_n[n].append(data)
    return by_n


def find_wing_images(ablation_dir, sample_idx=0):
    """Return dict: n_samples -> path to slice_comparison image for that size."""
    img_paths = {}
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s0"))):
        m = re.search(r"n(\d+)_s0", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        pngs = sorted(glob.glob(os.path.join(n_dir, f"eval_*_slice_comparison.png")))
        if pngs:
            img_paths[n] = pngs[-1]
    return img_paths


def find_wing3d_images(ablation_dir, sample_idx=0):
    """Return dict: n_samples -> path to the full 3D wing GT-vs-generated render for that size."""
    img_paths = {}
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s0"))):
        m = re.search(r"n(\d+)_s0", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        pngs = sorted(glob.glob(os.path.join(n_dir, f"eval_*_wing3d_s{sample_idx}.png")))
        if pngs:
            img_paths[n] = pngs[-1]
    return img_paths


def plot_metrics_curves(by_n, save_dir, tag):
    sizes = sorted(by_n.keys())
    available = [k for k in THESIS_METRICS if any(k in run for runs in by_n.values() for run in runs)]

    for metric_key in available:
        means, stds = [], []
        valid_sizes = []
        for n in sizes:
            vals = [run[metric_key] for run in by_n[n] if metric_key in run]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_sizes.append(n)
        if not valid_sizes:
            continue
        means = np.array(means)
        stds  = np.array(stds)

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(valid_sizes, means, marker="o", linewidth=1.8, color="steelblue")
        ax.fill_between(valid_sizes, means - stds, means + stds, alpha=0.25, color="steelblue")
        ax.set_xlabel("Training dataset size", fontsize=9)
        ax.set_ylabel(METRICS[metric_key], fontsize=9)
        ax.set_xticks(valid_sizes)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"ablation_{tag}_{metric_key}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def plot_wing_composite(img_paths, save_dir, tag, kind="slices"):
    """Save one full-resolution comparison image per training size into save_dir."""
    sizes = sorted(img_paths.keys())
    if not sizes:
        print(f"No {kind} images found, skipping composite.")
        return

    out_dir = os.path.join(save_dir, f"ablation_{tag}_{kind}")
    os.makedirs(out_dir, exist_ok=True)

    for size in sizes:
        img = mpimg.imread(img_paths[size])
        fig, ax = plt.subplots(figsize=(img.shape[1] / 150, img.shape[0] / 150))
        ax.imshow(img)
        ax.set_title(f"DDM_W3D Ablation — n={size} ({tag}, seed 0)", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{kind}_n{size:04d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", default="results/ddm_w_3d_ablation_v29")
    p.add_argument("--out_dir",      default="results")
    p.add_argument("--wing_sample",  type=int, default=0,
                   help="Which test sample index to use for the wing composite (default: 0)")
    p.add_argument("--tag", default=None,
                   help="Label used in output filenames/titles (default: derived from --ablation_dir)")
    args = p.parse_args()

    tag = args.tag or os.path.basename(os.path.normpath(args.ablation_dir)).replace("ddm_w_3d_ablation_", "")

    os.makedirs(args.out_dir, exist_ok=True)
    by_n = find_eval_jsons(args.ablation_dir)

    if not by_n:
        print(f"No eval JSON files found in {args.ablation_dir}. Run the ablation first.")
        return

    sizes = sorted(by_n.keys())
    print(f"Found data for n_samples: {sizes}")
    for n in sizes:
        print(f"  n={n}: {len(by_n[n])} seed(s)")

    plot_metrics_curves(by_n, args.out_dir, tag)

    img_paths = find_wing_images(args.ablation_dir, sample_idx=args.wing_sample)
    plot_wing_composite(img_paths, args.out_dir, tag, kind="slices")

    wing3d_paths = find_wing3d_images(args.ablation_dir, sample_idx=args.wing_sample)
    plot_wing_composite(wing3d_paths, args.out_dir, tag, kind="wing3d")


if __name__ == "__main__":
    main()
