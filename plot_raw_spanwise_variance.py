"""
Raw-data spanwise variance — shows where along the span the wing geometry
varies most across the dataset (root vs. tip), independent of any model.

For every wing in the dataset, computes per-slice coordinate variance
relative to the dataset mean shape at that span, then aggregates across
all wings. Motivates why root/tip are harder for the BAE: those spans
carry the most shape variability to begin with.

Usage
-----
    .venv/bin/python plot_raw_spanwise_variance.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--final_only", action="store_true", default=True)
    p.add_argument("--out", type=str, default="thesis/figures/raw_spanwise_variance.pdf")
    return p.parse_args()


def main():
    args = parse_args()

    dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_items = list(dataset["train"]) + list(dataset["val"]) + list(dataset["test"])
    if args.final_only:
        all_items = [item for item in all_items if item["final"] == 1]
    print(f"Wings: {len(all_items)}")

    coords = np.stack([item["coords"] for item in all_items])   # [N, S, 192, 2]
    eta = all_items[0]["transforms"]                              # [S]
    S = coords.shape[1]

    # Per-span variance across the dataset: mean over points/coords of the
    # per-point variance across wings (relative to the dataset's mean shape
    # at that span).
    mean_shape = coords.mean(axis=0, keepdims=True)               # [1, S, 192, 2]
    var_per_point = ((coords - mean_shape) ** 2).mean(axis=0)      # [S, 192, 2]
    var_per_span = var_per_point.mean(axis=(1, 2))                 # [S]
    std_per_span = np.sqrt(var_per_span)

    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.plot(eta, var_per_span, color="steelblue", marker="o", ms=4, lw=1.8)

    ax.set_xlabel(r"Span position $z$ (root = 0, tip $\approx$ 2.25)", fontsize=9)
    ax.set_ylabel("Mean coordinate variance across dataset", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    print(f"\n{'Span':>5}  {'eta':>8}  {'Variance':>12}  {'Std':>10}")
    for s in range(S):
        print(f"{s+1:>5}  {eta[s]:>8.3f}  {var_per_span[s]:>12.6f}  {std_per_span[s]:>10.6f}")


if __name__ == "__main__":
    main()
