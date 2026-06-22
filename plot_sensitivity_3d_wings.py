"""
Plot 3D views of the 10 generated wings per GT for the sensitivity analysis.

For each anchor (GT flow condition) in the sensitivity JSON, produces:
  - One figure with 11 subplots: GT wing + 10 generated wings, all as 3D surfaces.

Usage:
    python plot_sensitivity_3d_wings.py [--json PATH] [--out_dir DIR] [--anchors N]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# scienceplots (imported inside engiopt.ddm.plotting) sets usetex=True — override after
from engiopt.ddm.plotting import wing_3D_shape_plot
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["text.latex.preamble"] = ""
plt.rcParams["text.usetex"] = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        default="results/sensitivity_3d/sensitivity_ddm_w_20260610_094143.json",
    )
    p.add_argument("--out_dir", default="results/sensitivity_3d/wings_3d")
    p.add_argument("--anchors", type=int, default=None, help="Limit number of anchors plotted")
    p.add_argument("--wing_len", type=float, default=2.25)
    p.add_argument("--elev", type=float, default=25)
    p.add_argument("--azim", type=float, default=-60)
    return p.parse_args()


def plot_anchor(anchor, wing_len, elev, azim, out_path):
    coords_list = [np.array(c) for c in anchor["coords_list"]]  # list of (15,2,192)
    gt_coords   = np.array(anchor["gt_coords"])                  # (15,2,192)
    n_inits     = len(coords_list)

    n_cols = n_inits + 1  # GT + generated
    fig = plt.figure(figsize=(4 * n_cols, 5))

    z = np.linspace(0, wing_len, gt_coords.shape[0])

    # GT wing
    ax = fig.add_subplot(1, n_cols, 1, projection="3d")
    wing_3D_shape_plot(gt_coords, ax=ax, facecolor="coral", alpha=0.75,
                       z=z, wing_len=wing_len)
    ax.set_title("GT", fontsize=8, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # 10 generated wings
    for i, coords in enumerate(coords_list):
        ax = fig.add_subplot(1, n_cols, i + 2, projection="3d")
        wing_3D_shape_plot(coords, ax=ax, facecolor="steelblue", alpha=0.75,
                           z=z, wing_len=wing_len)
        ax.set_title(f"Gen {i+1}", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    case = anchor["case_num"]
    m    = anchor["mach"]
    re   = anchor["reynolds"]
    cl   = anchor["cl_target"]
    regime = anchor.get("regime", "")
    pw_mse = anchor["pairwise_shape_mse"]
    gt_mse = anchor["mean_gt_mse"]
    aoa_std = anchor["aoa_std"]

    fig.suptitle(
        f"Case {case}  |  M={m:.2f}  Re={re/1e6:.1f}M  CL={cl:.3f}  ({regime})\n"
        f"pairwise MSE={pw_mse:.2e}   mean GT-MSE={gt_mse:.2e}   AoA std={aoa_std:.3f}°",
        fontsize=9, y=1.01,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.05)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    with open(args.json) as f:
        data = json.load(f)

    anchors = data["per_anchor"]
    if args.anchors is not None:
        anchors = anchors[: args.anchors]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Plotting {len(anchors)} anchors → {args.out_dir}/")

    for anchor in anchors:
        case = anchor["case_num"]
        regime = anchor.get("regime", "unk")
        fname = f"case_{case}_{regime}_3d.png"
        plot_anchor(anchor, args.wing_len, args.elev, args.azim,
                    os.path.join(args.out_dir, fname))

    print("Done.")


if __name__ == "__main__":
    main()
