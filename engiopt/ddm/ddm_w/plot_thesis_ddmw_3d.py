"""
Thesis-quality 3D surface plots of DDM-W generated wings, coloured by predicted Cp.

One wing per flow regime (subsonic / transonic / supersonic), each saved as its own
PDF so they can be placed independently in the thesis.

Usage
-----
    python -m engiopt.ddm.ddm_w.plot_thesis_ddmw_3d
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
import matplotlib.cm as _cm
import matplotlib.colors as _mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "figure.dpi":     150,
})




def load_ddm_w(bae_model, lvae_model, device):
    ckpt = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    cfg  = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = device
    sampler = build_sampler(cfg)

    saved = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=cfg.lae_latent_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)

    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=cfg.lae_latent_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=ckpt.get("params_mean_std"),
        aoas_mean_std=ckpt.get("aoas_mean_std"),
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device).eval()
    return ddm_w, ckpt


def _draw_smooth_surface(ax, wing, pres, n_pts=128, span_chord_ratio=2.505):
    """Draw a continuous Cp-coloured 3D wing surface with no visible slice seams."""
    S, _, N = wing.shape
    cmap = _cm.get_cmap("RdBu_r")
    all_p = pres.ravel()
    pmin, pmax = float(np.nanpercentile(all_p, 2)), float(np.nanpercentile(all_p, 98))
    norm = _mcolors.Normalize(vmin=pmin, vmax=pmax)

    span_pos = np.linspace(0.0, span_chord_ratio, S)

    # Resample each slice to uniform n_pts on upper and lower surfaces separately
    upper_x = np.zeros((S, n_pts))
    upper_y = np.zeros((S, n_pts))
    upper_p = np.zeros((S, n_pts))
    lower_x = np.zeros((S, n_pts))
    lower_y = np.zeros((S, n_pts))
    lower_p = np.zeros((S, n_pts))

    for s in range(S):
        x, y, p = wing[s, 0], wing[s, 1], pres[s]
        le = int(np.argmin(x))
        # Upper: le → TE (index 0) going one way
        ui = np.arange(le, -1, -1) if le > N // 2 else np.arange(le, N)
        # Split cleanly: points with index <= le go upper, rest lower
        upper_idx = np.arange(0, le + 1)
        lower_idx = np.arange(le, N)
        for arr, idx, ux, uy, up in [
            (None, upper_idx, upper_x, upper_y, upper_p),
            (None, lower_idx, lower_x, lower_y, lower_p),
        ]:
            xi, yi, pi = x[idx], y[idx], p[idx]
            t0 = np.linspace(0, 1, len(idx))
            t1 = np.linspace(0, 1, n_pts)
            ux[s] = interp1d(t0, xi, kind="linear")(t1)
            uy[s] = interp1d(t0, yi, kind="linear")(t1)
            up[s] = interp1d(t0, pi, kind="linear")(t1)

    span_grid = np.tile(span_pos[:, None], (1, n_pts))

    for X, Z, P in [(upper_x, upper_y, upper_p), (lower_x, lower_y, lower_p)]:
        facecolors = cmap(norm(P))
        ax.plot_surface(X, span_grid, Z,
                        facecolors=facecolors,
                        alpha=1.0,
                        linewidth=0,
                        antialiased=True,
                        rcount=S,
                        ccount=n_pts)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "thesis/figures"
    os.makedirs(save_dir, exist_ok=True)

    bae_model  = load_bae_3d(BAE_CKP, device)
    cfg = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device = device
    lvae_model = load_lvae_3d(cfg, bae_model)
    ddm_w, ckpt = load_ddm_w(bae_model, lvae_model, device)

    dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_test     = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_final      = [it for it in all_test if it["final"] == 1]

    sc_params = scaler(ckpt["params_mean_std"]) if ckpt.get("params_mean_std") is not None else None
    sc_aoas   = scaler(ckpt["aoas_mean_std"])   if ckpt.get("aoas_mean_std")   is not None else None

    _, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
        test_final, initial_by_case, bae_model, lvae_model,
        ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, device,
    )

    machs = np.array([it["mach"] for it in test_final])
    regime_indices = {
        "subsonic":  int(np.where(machs < 0.8)[0][0]),
        "transonic": int(np.where((machs >= 0.8) & (machs < 1.0))[0][0]),
        "supersonic":int(np.where(machs >= 1.0)[0][0]),
    }
    print("Wing indices per regime:", regime_indices)

    for regime, idx in regime_indices.items():
        item = test_final[idx]
        mach = item["mach"]
        re   = item["reynolds"]
        cl   = item["cl_target"]
        ar   = item["area_case_ratio"]
        print(f"Generating {regime} wing (index {idx}, M={mach:.3f}) ...")
        with torch.no_grad():
            gen_coords, gen_aoas, gen_pressures, _, _, _ = ddm_w.generate(
                w_init=w_inits[idx:idx+1].to(device),
                params=params_norm[idx:idx+1].to(device),
                device=device,
            )

        wing = gen_coords[0].cpu().numpy()       # [S, 2, 192]
        pres = gen_pressures[0].cpu().numpy()    # [S, 192]
        aoa  = float(gen_aoas[0].cpu())

        fig = plt.figure(figsize=(10, 4.5))
        ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
        fig.subplots_adjust(left=-0.05, right=0.85, top=1.05, bottom=-0.08)

        _draw_smooth_surface(ax, wing, pres)

        # Span is 2.505, chord is 1 — reflect true proportions
        ax.set_box_aspect([1, 2.505, 0.18])
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel("x/c", labelpad=8)
        ax.set_ylabel("Span η", labelpad=20)
        ax.set_zlabel("y/c", labelpad=14)
        ax.set_zticks(np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 3))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linewidth=0.3, alpha=0.4)

        # Fewer tick labels on span axis
        ax.set_yticks(np.linspace(0, 2.505, 5))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 2.505, 5)])

        # Colorbar pushed right so it doesn't overlap y/c label
        sm = _cm.ScalarMappable(cmap="RdBu_r",
                                norm=_mcolors.Normalize(
                                    vmin=pres.min(), vmax=pres.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.15, aspect=18)
        cbar.set_label("$C_p$", fontsize=9)

        # Flow condition annotation at top — two lines for readability
        cond_line1 = (f"$M_\\infty={mach:.2f}$   $Re={re/1e6:.2f}\\times10^6$   "
                      f"$C_L^{{\\rm target}}={cl:.2f}$")
        cond_line2 = f"$AR={ar:.2f}$   $\\alpha={aoa:.1f}^\\circ$"
        fig.text(0.18, 0.95, cond_line1, ha="left", va="top", fontsize=12,
                 transform=fig.transFigure)
        fig.text(0.18, 0.88, cond_line2, ha="left", va="top", fontsize=12,
                 transform=fig.transFigure)

        for ext in ("pdf", "png"):
            path = os.path.join(save_dir, f"ddmw_3d_{regime}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved: {path}")

        # Crop whitespace from PNG using PIL
        png_path = os.path.join(save_dir, f"ddmw_3d_{regime}.png")
        from PIL import Image, ImageChops
        img = Image.open(png_path).convert("RGB")
        bg  = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            pad = 10  # pixels
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(img.width,  bbox[2] + pad),
                    min(img.height, bbox[3] + pad))
            img.crop(bbox).save(png_path)
            print(f"Cropped: {png_path}")

        plt.close(fig)

        # 2-D Cp vs x/c at mid-span for all regimes
        if True:
            mid = wing.shape[0] // 2          # mid-span slice index
            x_coords = wing[mid, 0]           # [192]
            y_coords = wing[mid, 1]           # [192]
            cp_mid   = pres[mid]              # [192]

            # Split into upper and lower surface at the leading edge (min x)
            le = int(np.argmin(x_coords))
            upper_idx = np.arange(0, le + 1)
            lower_idx = np.arange(le, len(x_coords))

            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            ax2.plot(x_coords[upper_idx], cp_mid[upper_idx],
                     color="steelblue", lw=1.5, label="Upper surface")
            ax2.plot(x_coords[lower_idx], cp_mid[lower_idx],
                     color="firebrick", lw=1.5, linestyle="--", label="Lower surface")
            ax2.invert_yaxis()   # aerodynamic convention: -Cp up
            ax2.set_xlabel("$x/c$")
            ax2.set_ylabel("$C_p$")
            ax2.set_xlim(0, 1)
            ax2.legend(fontsize=8)
            ax2.grid(True, linewidth=0.4, alpha=0.5)
            fig2.tight_layout()
            cp_path = os.path.join(save_dir, f"ddmw_cp_{regime}_midspan.pdf")
            fig2.savefig(cp_path, bbox_inches="tight")
            print(f"Saved: {cp_path}")
            plt.close(fig2)

    print("Done.")


if __name__ == "__main__":
    main()
