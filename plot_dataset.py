"""
Dataset visualisation figures.

Usage:
    .venv/bin/python plot_dataset.py              # 2D slice grid
    .venv/bin/python plot_dataset.py --mode 3d    # 3D surface via BAE roundtrip
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT = "thesis/figures"
os.makedirs(OUT, exist_ok=True)

CASE_NUM  = 40
ROWS, COLS = 5, 3

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


# ── 2D grid ───────────────────────────────────────────────────────────────────

def make_slice_grid(df, out_path):
    df_case = df[df["case_num"] == float(CASE_NUM)]
    slice_nums    = sorted(df_case["slice_num"].unique())
    slice_initial = int(min(slice_nums))
    slice_final   = int(max(slice_nums))

    df_init  = df_case[df_case["slice_num"] == slice_initial]
    df_final = df_case[df_case["slice_num"] == slice_final]

    sub_slices = sorted(df_init["sub_slice_num"].unique())
    etas       = [df_init[df_init["sub_slice_num"] == s]["eta"].iloc[0] for s in sub_slices]
    assert len(sub_slices) == 15, f"Expected 15 sub-slices, got {len(sub_slices)}"

    slice_ylims = {}
    for ss in sub_slices:
        y = np.concatenate([
            df_init[df_init["sub_slice_num"] == ss]["CoordinateY"].values,
            df_final[df_final["sub_slice_num"] == ss]["CoordinateY"].values,
        ])
        spread = y.max() - y.min()
        slice_ylims[ss] = (y.min() - spread * 0.1, y.max() + spread * 0.1)

    fig = plt.figure(figsize=(14, 7.5))
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              left=0.06, right=0.98, top=0.96, bottom=0.12, wspace=0.14)

    for p, (df_panel, color) in enumerate(zip([df_init, df_final],
                                               ["#2166ac", "#d6604d"])):
        inner = gridspec.GridSpecFromSubplotSpec(ROWS, COLS, subplot_spec=outer[p],
                                                  hspace=0.15, wspace=0.18)
        for i, (ss, eta) in enumerate(zip(sub_slices, etas)):
            row, col = divmod(i, COLS)
            ax = fig.add_subplot(inner[row, col])
            df_s = df_panel[df_panel["sub_slice_num"] == ss]
            ax.plot(df_s["CoordinateX"], df_s["CoordinateY"], "-", color=color, linewidth=0.9)
            ax.set_title(f"$\\eta = {eta:.2f}$", pad=3, fontsize=13)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(slice_ylims[ss])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax.tick_params(labelsize=12, length=2)
            if row < ROWS - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

    fig.text(0.29, 0.10, r"$x/c$ (normalised chord)", ha="center", fontsize=14)
    fig.text(0.75, 0.10, r"$x/c$ (normalised chord)", ha="center", fontsize=14)
    fig.text(0.015, 0.47, r"$y/c$ (normalised thickness)", va="center",
             rotation="vertical", fontsize=14)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_2d():
    print("Loading uniform dataset...")
    with open("Wing_TL/data/processed/new_dataset_slices.pkl", "rb") as f:
        df_uniform = pickle.load(f)
    make_slice_grid(df_uniform, f"{OUT}/dataset2_case40_initial_final.pdf")

    custom_path = "Wing_TL/data/processed/new_dataset_custom_etas_slices.pkl"
    if os.path.exists(custom_path):
        print("Loading custom-eta dataset...")
        with open(custom_path, "rb") as f:
            df_custom = pickle.load(f)
        make_slice_grid(df_custom, f"{OUT}/dataset2_case40_custom_etas_initial_final.pdf")


# ── 3D surfaces ───────────────────────────────────────────────────────────────

def plot_3d():
    import torch
    from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
    from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D

    _CUSTOM_ETAS_PKL = "Wing_TL/data/processed/new_dataset_custom_etas_slices.pkl"
    _UNIFORM_PKL     = "Wing_TL/data/processed/new_dataset_slices.pkl"
    _SCALARS_PKL     = "Wing_TL/data/processed/new_dataset_scalars.pkl"
    _BAE_CKPT        = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
    _SPAN_CHORD_RATIO = 2.505

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi":     150,
    })

    def load_case(df, case_num, is_final):
        case   = df[df["case_num"] == case_num]
        slices = sorted(case["slice_num"].unique())
        sn     = max(slices) if is_final else min(slices)
        df_wing = case[case["slice_num"] == sn]
        sub_slices = sorted(df_wing["sub_slice_num"].unique())
        coords = np.stack([
            np.stack([
                df_wing[df_wing["sub_slice_num"] == ss]["CoordinateX"].values,
                df_wing[df_wing["sub_slice_num"] == ss]["CoordinateY"].values,
            ], axis=1)
            for ss in sub_slices
        ])
        etas = [df_wing[df_wing["sub_slice_num"] == ss]["eta"].iloc[0] for ss in sub_slices]
        return coords, etas

    def bae_roundtrip(coords_np, model, device):
        item    = {"coords": coords_np, "te_shifts": np.zeros(coords_np.shape[0], dtype=np.float32)}
        dataset = WingsBezierDataset3D([item], num_extra_tip_slices=0)
        x = dataset[0].unsqueeze(0).to(device)
        with torch.no_grad():
            z = model.encode(x)
            y, _ = model.decode(z, return_cp=True)
        return y.squeeze(0).cpu().numpy()

    def draw_surface(ax, data):
        S, _, N = data.shape
        span_pos = np.linspace(0.0, 1.0, S)
        X = data[:, 0, :]
        Y = np.tile(span_pos[:, None], (1, N))
        Z = data[:, 1, :]
        surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.72,
                               linewidth=0, antialiased=True,
                               rcount=S, ccount=64,
                               vmin=Z.min(), vmax=Z.max())
        cmap    = plt.get_cmap("coolwarm")
        colours = [cmap(i / (S - 1)) for i in range(S)]
        for s in range(S):
            ax.plot(data[s, 0], np.full(N, span_pos[s]), data[s, 1],
                    color=colours[s], lw=0.8, alpha=0.6)
        ax.set_xlabel("x/c", labelpad=6)
        ax.set_ylabel("Span η", labelpad=6)
        ax.set_zlabel("y/c", labelpad=6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(-0.55, 0.55)
        ax.set_box_aspect([1, _SPAN_CHORD_RATIO, 1])
        ax.view_init(elev=22, azim=-55)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linewidth=0.3, alpha=0.4)
        return surf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(_BAE_CKPT, map_location=device, weights_only=False)
    model  = BezierAutoencoder3D(
        n_spans=ckpt["n_spans"],
        n_control_points=ckpt["n_control_points"],
        n_data_points=192,
        slice_hidden_dims=ckpt["slice_hidden_dims"],
        span_hidden_dims=ckpt["span_hidden_dims"],
        latent_dim=ckpt["latent_dim"],
        cpx_bound=ckpt["cpx_bound"],
        cpy_bound=ckpt["cpy_bound"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded BAE (latent_dim={ckpt['latent_dim']}, n_spans={ckpt['n_spans']})")

    datasets = {}
    if os.path.exists(_CUSTOM_ETAS_PKL):
        datasets["custom_etas"] = _CUSTOM_ETAS_PKL
    datasets["uniform"] = _UNIFORM_PKL

    for ds_name, pkl_path in datasets.items():
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        for is_final, suffix in [(False, "initial"), (True, "final")]:
            coords_np, _ = load_case(df, float(CASE_NUM), is_final)
            recon = bae_roundtrip(coords_np, model, device)
            fig   = plt.figure(figsize=(8, 5))
            ax    = fig.add_subplot(111, projection="3d")
            surf  = draw_surface(ax, recon)
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, aspect=20,
                         label="y/c  (surface height)")
            out = f"{OUT}/dataset2_case40_3d_{ds_name}_{suffix}.pdf"
            fig.savefig(out, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved: {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["2d", "3d"], default="2d",
                   help="2d = spanwise slice grid; 3d = BAE roundtrip surface")
    args = p.parse_args()
    if args.mode == "2d":
        plot_2d()
    else:
        plot_3d()


if __name__ == "__main__":
    main()
