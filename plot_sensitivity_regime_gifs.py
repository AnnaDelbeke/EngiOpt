"""
Rotating GIFs of the sensitivity analysis results, grouped by flow regime.

For each regime (subsonic / transonic / supersonic) one GIF is produced showing
all anchors in that regime as rows.  Each row has 11 panels: GT wing (coral) +
10 generated wings (steelblue), all rotating synchronously.

Layout example for subsonic (5 anchors):
    row 0  GT | Gen1 | Gen2 | … | Gen10
    row 1  GT | Gen1 | …
    …

Usage:
    python plot_sensitivity_regime_gifs.py
    python plot_sensitivity_regime_gifs.py \\
        --json results/sensitivity_3d/sensitivity_ddm_w_20260610_094143.json \\
        --out_dir results/sensitivity_3d/regime_gifs \\
        --n_frames 36 --fps 12
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# scienceplots sets usetex=True — kill it immediately after import
from engiopt.ddm.plotting import wing_3D_shape_plot
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["text.latex.preamble"] = ""
plt.rcParams["text.usetex"] = False

_PANEL_W   = 3.0   # inches per panel
_PANEL_H   = 3.2   # inches per row
_WING_LEN  = 2.25
_ELEV      = 25
_REGIME_COLOURS = {
    "subsonic":    "steelblue",
    "transonic":   "seagreen",
    "supersonic":  "darkorange",
}


def _draw_wing_panel(ax, coords, title, facecolor, elev, azim, wing_len):
    z = np.linspace(0, wing_len, coords.shape[0])
    wing_3D_shape_plot(coords, ax=ax, facecolor=facecolor, alpha=0.75,
                       z=z, wing_len=wing_len)
    ax.set_title(title, fontsize=6, pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def make_regime_gif(anchors, regime, out_path,
                    n_frames=36, fps=12,
                    elev=_ELEV, wing_len=_WING_LEN):
    n_rows = len(anchors)
    n_cols = 11                        # GT + 10 generated
    gen_colour = _REGIME_COLOURS.get(regime, "steelblue")

    fig_w = _PANEL_W * n_cols
    fig_h = _PANEL_H * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = []   # list of lists: axes[row][col]
    for row, anchor in enumerate(anchors):
        gt_coords   = np.array(anchor["gt_coords"])    # (15,2,192)
        coords_list = [np.array(c) for c in anchor["coords_list"]]

        row_axes = []
        # GT panel
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 1, projection="3d")
        case = anchor["case_num"]
        m    = anchor["mach"]
        re   = anchor["reynolds"] / 1e6
        cl   = anchor["cl_target"]
        _draw_wing_panel(ax, gt_coords, f"GT  C{case}\nM{m:.2f} Re{re:.1f}M CL{cl:.3f}",
                         "coral", elev, -60, wing_len)
        row_axes.append(ax)

        # Generated panels
        for i, coords in enumerate(coords_list):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + i + 2, projection="3d")
            _draw_wing_panel(ax, coords, f"Gen {i+1}", gen_colour, elev, -60, wing_len)
            row_axes.append(ax)

        axes.append(row_axes)

    fig.suptitle(f"Initialisation sensitivity — {regime.capitalize()}",
                 fontsize=10, y=1.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95,
                        bottom=0.005, wspace=0.02, hspace=0.12)

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    def update(frame):
        az = azimuths[frame]
        for row_axes in axes:
            for ax in row_axes:
                ax.view_init(elev=elev, azim=az)
        return []

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000 // fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps, dpi=90)
    plt.close(fig)
    print(f"  saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json",
                   default="results/sensitivity_3d/sensitivity_ddm_w_20260610_094143.json")
    p.add_argument("--out_dir", default="results/sensitivity_3d/regime_gifs")
    p.add_argument("--n_frames", type=int, default=36)
    p.add_argument("--fps",      type=int, default=12)
    p.add_argument("--elev",     type=float, default=_ELEV)
    p.add_argument("--wing_len", type=float, default=_WING_LEN)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.json) as f:
        data = json.load(f)

    by_regime = defaultdict(list)
    for anchor in data["per_anchor"]:
        by_regime[anchor["regime"]].append(anchor)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Regimes found: { {r: len(v) for r, v in by_regime.items()} }")
    print(f"Output dir: {args.out_dir}/\n")

    for regime, anchors in by_regime.items():
        print(f"── {regime} ({len(anchors)} anchors) ──")
        out_path = os.path.join(args.out_dir, f"sensitivity_{regime}.gif")
        make_regime_gif(anchors, regime, out_path,
                        n_frames=args.n_frames, fps=args.fps,
                        elev=args.elev, wing_len=args.wing_len)

    print("\nDone.")


if __name__ == "__main__":
    main()
