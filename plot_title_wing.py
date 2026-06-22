"""
Title picture: swept/tapered 3D wing surface coloured by Cp.
Dark background, no axes, no labels, no colorbar.

Usage:
    python plot_title_wing.py [--out results/title_wing.png]
"""

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d


_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_50slices_slices.pkl"
_SPAN_CHORD_RATIO = 2.505
_N_PTS = 192


def load_wing(df, case_num, slice_num=None):
    """Return coords [S,2,N] and Cp [S,N] for one wing snapshot."""
    case = df[df["case_num"] == case_num]
    if slice_num is None:
        # pick the last optimisation slice (final wing)
        slice_num = sorted(case["slice_num"].unique())[-1]
    snap = case[case["slice_num"] == slice_num]
    sub_slices = sorted(snap["sub_slice_num"].unique())

    coords_list, cp_list = [], []
    for ss in sub_slices:
        rows = snap[snap["sub_slice_num"] == ss]
        coords_list.append(np.stack([rows["CoordinateX"].values,
                                      rows["CoordinateY"].values], axis=1))  # [N,2]
        cp_list.append(rows["CoefPressure"].values)  # [N]

    coords = np.stack(coords_list)   # [S, N, 2]
    coords = coords.transpose(0, 2, 1)  # [S, 2, N]
    cp     = np.stack(cp_list)           # [S, N]
    return coords, cp


def draw_cp_surface(ax, wing, cp, pmin, pmax, n_pts=_N_PTS, span_chord_ratio=_SPAN_CHORD_RATIO):
    """Plot swept, tapered wing surface coloured by Cp.

    wing: [S, 2, N]   (x/c, y/c coordinates per spanwise station)
    cp:   [S, N]
    """
    S, _, N = wing.shape

    # Apply sweep and taper to x-positions so the wing looks planform-correct.
    # Root eta=0, tip eta=1 (larger span position).
    eta = np.linspace(0.0, 1.0, S)
    sweep_deg = 28.0          # leading-edge sweep
    sweep_tan = np.tan(np.deg2rad(sweep_deg))
    taper     = 0.38          # tip/root chord ratio

    cmap = cm.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=pmin, vmax=pmax)

    # Span positions in physical units (scaled by span/chord ratio)
    span_pos = eta * span_chord_ratio  # [S]

    upper_X = np.zeros((S, n_pts))
    upper_Y = np.zeros((S, n_pts))
    upper_Z = np.zeros((S, n_pts))
    upper_P = np.zeros((S, n_pts))
    lower_X = np.zeros((S, n_pts))
    lower_Y = np.zeros((S, n_pts))
    lower_Z = np.zeros((S, n_pts))
    lower_P = np.zeros((S, n_pts))

    for s in range(S):
        chord_s = 1.0 - (1.0 - taper) * eta[s]   # normalised chord at this station
        le_x    = sweep_tan * eta[s] * span_chord_ratio  # leading-edge x-offset
        x_raw = wing[s, 0]   # [N]  in [0,1]
        y_raw = wing[s, 1]   # [N]
        p_raw = cp[s]        # [N]

        # Find leading-edge index (minimum x)
        le_idx = int(np.argmin(x_raw))

        for indices, Xbuf, Zbuf, Pbuf in [
            (np.arange(0, le_idx + 1), upper_X, upper_Z, upper_P),
            (np.arange(le_idx, N),     lower_X, lower_Z, lower_P),
        ]:
            t0 = np.linspace(0, 1, len(indices))
            t1 = np.linspace(0, 1, n_pts)
            xi  = interp1d(t0, x_raw[indices], kind="linear")(t1)
            yi  = interp1d(t0, y_raw[indices], kind="linear")(t1)
            pi  = interp1d(t0, p_raw[indices], kind="linear")(t1)
            # Apply sweep (x-offset) and taper (chord scaling)
            Xbuf[s]  = le_x + xi * chord_s
            Zbuf[s]  = yi * chord_s
            Pbuf[s]  = pi

        upper_Y[s] = span_pos[s]
        lower_Y[s] = span_pos[s]

    for X, Y, Z, P in [(upper_X, upper_Y, upper_Z, upper_P),
                        (lower_X, lower_Y, lower_Z, lower_P)]:
        ax.plot_surface(X, Y, Z,
                        facecolors=cmap(norm(P)),
                        alpha=1.0, linewidth=0, antialiased=True,
                        rcount=S, ccount=n_pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/title_wing.png")
    parser.add_argument("--case_num", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--span_chord_ratio", type=float, default=_SPAN_CHORD_RATIO)
    parser.add_argument("--pmin", type=float, default=None, help="Cp colormap min")
    parser.add_argument("--pmax", type=float, default=None, help="Cp colormap max")
    args = parser.parse_args()

    print("Loading data…")
    with open(_SLICES_PKL, "rb") as f:
        df = pickle.load(f)

    case_num = args.case_num
    if case_num is None:
        case_num = float(sorted(df["case_num"].unique())[0])
    print(f"Using case_num={case_num}")

    wing, cp = load_wing(df, case_num)
    print(f"Wing shape: {wing.shape}, Cp range: [{cp.min():.3f}, {cp.max():.3f}]")

    pmin = args.pmin if args.pmin is not None else max(cp.min(), -1.2)
    pmax = args.pmax if args.pmax is not None else min(cp.max(),  0.5)
    scr  = args.span_chord_ratio

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    draw_cp_surface(ax, wing, cp, pmin=pmin, pmax=pmax, span_chord_ratio=scr)

    ax.view_init(elev=18, azim=-50)

    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color("none")
    ax.yaxis.line.set_color("none")
    ax.zaxis.line.set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    sweep_tan = np.tan(np.deg2rad(28.0))
    max_x = sweep_tan * scr + 1.0
    ax.set_xlim(-0.05, max_x + 0.05)
    ax.set_ylim(-0.05, scr + 0.05)
    ax.set_zlim(-0.20, 0.25)
    ax.set_box_aspect([max_x / scr, 1.0, 0.18])

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
