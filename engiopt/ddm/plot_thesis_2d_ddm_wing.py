"""Render a 2D-DDM generated wing overlaid with its paired ground-truth wing
as a clean 3D cage plot for the thesis.

Usage:
    python -m engiopt.ddm.plot_thesis_2d_ddm_wing [wing_index]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops

plt.rcParams["text.usetex"] = False
# engiopt.ddm's package __init__ imports plotting.py, which applies the
# scienceplots "science" style; that style sets savefig.bbox="tight", which
# silently re-crops this figure on save regardless of the bbox_inches passed
# to savefig(), clipping the manually-placed axis labels. Disable it here.
plt.rcParams["savefig.bbox"] = None

DATA_DIR = "results/generated/20260401_153629_v7hope250:1"
OUT_DIR = "thesis/figures"
WING_LEN = 2.25
N_DISPLAY_SLICES = 5


def plot_wing_slices(wing, ax, color, n_display_slices, label=None):
    """Plot one wing as stacked 2D slices. wing: [n_slices, 2, n_points]."""
    n_slices = wing.shape[0]
    z_positions = np.linspace(0, WING_LEN, n_slices)
    display_idx = np.linspace(0, n_slices - 1, n_display_slices).round().astype(int)

    for i, s in enumerate(display_idx):
        x, y = wing[s, 0], wing[s, 1]
        ax.plot(
            x, np.full_like(x, z_positions[s]), y, "-", color=color, linewidth=1.6, zorder=5,
            label=label if i == 0 else None,
        )
    return z_positions


def style_axes(ax, x_lo, x_hi, y_lo, y_hi):
    ax.set_xlabel("x/c", labelpad=15)
    ax.set_ylabel("z (span)", labelpad=15)
    # set_zlabel's automatic placement is unreliable at this view angle (it can
    # land off-frame), so the "y" label is added manually next to the axis
    # instead (see add_y_axis_label below).
    ax.set_xlim(x_lo - 0.05, x_hi + 0.05)
    ax.set_ylim(0, 2.50)
    ax.set_zlim(y_lo, y_hi)
    ax.set_yticks([0, WING_LEN / 2, WING_LEN])
    ax.yaxis.set_tick_params(pad=5)
    # Three z-ticks: min, mid, max.
    z_mid = 0.0 if y_lo < 0 < y_hi else (y_lo + y_hi) / 2
    ax.set_zticks(sorted({round(y_lo, 2), round(z_mid, 2), round(y_hi, 2)}))
    ax.set_box_aspect([1.2, 1, max(0.3, (y_hi - y_lo) / 1.2)])
    ax.view_init(elev=18, azim=-75)

    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.grid(False)


def add_y_axis_label(ax, x_hi):
    """Place a "y" text label to the right of the "0.00" vertical-axis tick.

    Uses the 2D screen-space projection of the (x_hi, WING_LEN, 0) data point
    (where the "0.00" tick renders) plus a fixed pixel offset, since 3D data
    coordinates get clipped at the axes limits and set_zlabel's placement is
    unreliable at this view angle.
    """
    x2d, y2d, _ = proj3d.proj_transform(x_hi, WING_LEN, 0, ax.get_proj())
    ax.annotate(
        "y",
        xy=(x2d, y2d),
        xytext=(55, 0),
        textcoords="offset points",
        fontsize=11,
        ha="left",
        va="center",
        xycoords="data",
        annotation_clip=False,
    )


def main():
    wing_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    src = os.path.join(DATA_DIR, f"wing{wing_idx:02d}_with_gt.pt")
    data = torch.load(src, map_location="cpu")
    gen = data["generated_airfoil"].numpy()                     # [9, 2, 192]
    gt = data["gt_coords"].numpy().transpose(0, 2, 1)            # [9, 192, 2] -> [9, 2, 192]
    alpha_gen = data["generated_alpha"]
    alpha_gt = data["gt_alpha"]
    mach = data["mach"]
    reynolds = data["reynolds"]
    cl_target = data["cl_target"]

    fig = plt.figure(figsize=(9.5, 6), dpi=300)
    ax = fig.add_subplot(projection="3d")

    z_positions = plot_wing_slices(gt, ax, color="tab:blue", n_display_slices=N_DISPLAY_SLICES, label="Ground Truth")
    plot_wing_slices(gen, ax, color="tab:red", n_display_slices=N_DISPLAY_SLICES, label="Generated")

    x_lo = min(gt[:, 0].min(), gen[:, 0].min())
    x_hi = max(gt[:, 0].max(), gen[:, 0].max())
    y_margin = 0.05
    y_lo = min(gt[:, 1].min(), gen[:, 1].min()) - y_margin
    y_hi = max(gt[:, 1].max(), gen[:, 1].max()) + y_margin
    style_axes(ax, x_lo, x_hi, y_lo, y_hi)
    add_y_axis_label(ax, x_hi)

    # Shaded ground plane at the chord line (y=0), for depth cues.
    plane = [[
        [x_lo, 0, 0], [x_hi, 0, 0],
        [x_hi, WING_LEN, 0], [x_lo, WING_LEN, 0],
    ]]
    ax.add_collection3d(Poly3DCollection(plane, facecolor="lightsteelblue", alpha=0.3, edgecolor="none", zorder=1))

    ax.set_title(
        rf"$M_\infty$={mach:.2f}, $Re$={reynolds:.1e}, $C_L$={cl_target:.2f}"
        "\n"
        rf"$\alpha_{{data}}$={alpha_gt:.2f}$^\circ$, $\alpha_{{gen}}$={alpha_gen:.2f}$^\circ$",
        fontsize=10,
        y=0.95,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.78, 0.85), fontsize=9, frameon=False)

    fig.subplots_adjust(left=0.02, right=0.65, top=0.85, bottom=0.05)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "" if wing_idx == 0 else f"_{wing_idx:02d}"
    out_path = os.path.join(OUT_DIR, f"ddm2d_generated_wing_cage{suffix}.png")
    plt.savefig(out_path)
    plt.close()
    crop_whitespace(out_path, margin=15)
    print(f"Saved {out_path}")


def crop_whitespace(path, margin=10):
    """Trim uniform white border left over from the wide canvas, since
    matplotlib's own tight-bbox cropping is unreliable for 3D axes here."""
    im = Image.open(path)
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(im.width, x1 + margin), min(im.height, y1 + margin)
    im.crop((x0, y0, x1, y1)).save(path)


if __name__ == "__main__":
    main()
