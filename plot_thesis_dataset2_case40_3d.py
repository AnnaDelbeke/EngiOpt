"""
Thesis figure: Case 40 (custom-eta sampling) initial and final wings rendered
as 3D surfaces after a BAE encode-decode roundtrip.

Produces two separate PDFs (initial / final) so LaTeX can label them (a)/(b).

Usage:
    python plot_thesis_dataset2_case40_3d.py
"""

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D

_CUSTOM_ETAS_PKL = "Wing_TL/data/processed/new_dataset_custom_etas_slices.pkl"
_UNIFORM_PKL     = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL     = "Wing_TL/data/processed/new_dataset_scalars.pkl"
_BAE_CKPT        = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
_CASE_NUM        = 40.0
_SPAN_CHORD_RATIO = 2.505

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":     150,
})


def _load_case(df, case_num, is_final):
    case = df[df["case_num"] == case_num]
    slices = sorted(case["slice_num"].unique())
    slice_num = max(slices) if is_final else min(slices)
    df_wing = case[case["slice_num"] == slice_num]
    sub_slices = sorted(df_wing["sub_slice_num"].unique())

    coords = np.stack([
        np.stack([
            df_wing[df_wing["sub_slice_num"] == ss]["CoordinateX"].values,
            df_wing[df_wing["sub_slice_num"] == ss]["CoordinateY"].values,
        ], axis=1)  # [192, 2]
        for ss in sub_slices
    ])  # [S, 192, 2]

    etas = [df_wing[df_wing["sub_slice_num"] == ss]["eta"].iloc[0] for ss in sub_slices]
    return coords, etas


def _bae_roundtrip(coords_np, model, device):
    """coords_np: [S, 192, 2] -> recon: [S, 2, 192]"""
    item = {"coords": coords_np, "te_shifts": np.zeros(coords_np.shape[0], dtype=np.float32)}
    dataset = WingsBezierDataset3D([item], num_extra_tip_slices=0)
    x = dataset[0].unsqueeze(0).to(device)  # [1, S, 2, 192]
    with torch.no_grad():
        z = model.encode(x)
        y, _ = model.decode(z, return_cp=True)
    return y.squeeze(0).cpu().numpy()  # [S, 2, 192]


def _draw_surface(ax, data):
    """data: [S, 2, N]"""
    S, _, N = data.shape
    span_pos = np.linspace(0.0, 1.0, S)
    X = data[:, 0, :]
    Y = np.tile(span_pos[:, None], (1, N))
    Z = data[:, 1, :]

    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.72,
                           linewidth=0, antialiased=True,
                           rcount=S, ccount=64,
                           vmin=Z.min(), vmax=Z.max())
    cmap = plt.get_cmap("coolwarm")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        "custom_etas": _CUSTOM_ETAS_PKL,
        "uniform":     _UNIFORM_PKL,
    }

    ckpt = torch.load(_BAE_CKPT, map_location=device, weights_only=False)
    model = BezierAutoencoder3D(
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

    for ds_name, pkl_path in datasets.items():
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)

        for is_final, suffix in [
            (False, "initial"),
            (True,  "final"),
        ]:
            coords_np, etas = _load_case(df, _CASE_NUM, is_final)
            recon = _bae_roundtrip(coords_np, model, device)

            fig = plt.figure(figsize=(8, 5))
            ax  = fig.add_subplot(111, projection="3d")
            surf = _draw_surface(ax, recon)
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, aspect=20,
                         label="y/c  (surface height)")

            out = f"thesis/figures/dataset2_case40_3d_{ds_name}_{suffix}.pdf"
            fig.savefig(out, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
