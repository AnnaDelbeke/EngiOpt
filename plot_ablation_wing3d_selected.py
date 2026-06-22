"""
Compose a 4-panel thesis figure from the DDM_W3D ablation wing3d images.
Crops the generated (right) half of each source image and arranges them
in a single row with n= labels.

Usage:
    python plot_ablation_wing3d_selected.py \
        [--ablation_dir results/ablation_v29_wing3d] \
        [--out results/ablation_wing3d_selected.pdf]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SELECTED = [50, 200, 500, 767]


def crop_right_half(img: np.ndarray) -> np.ndarray:
    """Return the right half of the image (the generated wing)."""
    w = img.shape[1]
    return img[:, w // 2:, :]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", default="results/ablation_v29_wing3d")
    p.add_argument("--out", default="results/ablation_wing3d_selected.pdf")
    return p.parse_args()


def main():
    args = parse_args()

    imgs = {}
    for n in SELECTED:
        path = os.path.join(args.ablation_dir, f"wing3d_n{n:04d}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        imgs[n] = crop_right_half(mpimg.imread(path))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, n in zip(axes, SELECTED):
        ax.imshow(imgs[n])
        ax.set_title(f"$n = {n}$", fontsize=11)
        ax.axis("off")

    fig.suptitle("DDM\\_W3D — generated wings vs training set size", fontsize=12, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
