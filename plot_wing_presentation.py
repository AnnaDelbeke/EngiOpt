"""
Generate a clean 3D wing plot for the thesis presentation slide.
No flow-condition text annotations; wing + colorbar only.

Usage
-----
    python plot_wing_presentation.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import matplotlib.colors as _mcolors
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d

from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
BAE_CKP      = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP     = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP     = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "figure.dpi":  200,
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


def draw_smooth_surface(ax, wing, pres, n_pts=128, span_chord_ratio=2.505):
    S, _, N = wing.shape
    cmap = _cm.get_cmap("RdBu_r")
    all_p = pres.ravel()
    pmin, pmax = float(np.nanpercentile(all_p, 2)), float(np.nanpercentile(all_p, 98))
    norm = _mcolors.Normalize(vmin=pmin, vmax=pmax)

    span_pos = np.linspace(0.0, span_chord_ratio, S)

    upper_x = np.zeros((S, n_pts))
    upper_y = np.zeros((S, n_pts))
    upper_p = np.zeros((S, n_pts))
    lower_x = np.zeros((S, n_pts))
    lower_y = np.zeros((S, n_pts))
    lower_p = np.zeros((S, n_pts))

    for s in range(S):
        x, y, p = wing[s, 0], wing[s, 1], pres[s]
        le = int(np.argmin(x))
        upper_idx = np.arange(0, le + 1)
        lower_idx = np.arange(le, N)
        for idx, ux, uy, up in [
            (upper_idx, upper_x, upper_y, upper_p),
            (lower_idx, lower_x, lower_y, lower_p),
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
                        alpha=1.0, linewidth=0, antialiased=True,
                        rcount=S, ccount=n_pts)

    return norm


def main():
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "thesis/figures"
    os.makedirs(save_dir, exist_ok=True)

    bae_model  = load_bae_3d(BAE_CKP, device)
    cfg = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device = device
    lvae_model = load_lvae_3d(cfg, bae_model)
    ddm_w, ckpt = load_ddm_w(bae_model, lvae_model, device)

    dataset         = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_test        = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_final      = [it for it in all_test if it["final"] == 1]

    sc_params = scaler(ckpt["params_mean_std"]) if ckpt.get("params_mean_std") is not None else None
    sc_aoas   = scaler(ckpt["aoas_mean_std"])   if ckpt.get("aoas_mean_std")   is not None else None

    _, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
        test_final, initial_by_case, bae_model, lvae_model,
        ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, device,
    )

    machs = np.array([it["mach"] for it in test_final])
    idx   = int(np.where(machs < 0.8)[0][0])   # first subsonic wing

    item = test_final[idx]
    print(f"Subsonic wing: index={idx}, M={item['mach']:.3f}, Re={item['reynolds']:.2e}")

    with torch.no_grad():
        gen_coords, gen_aoas, gen_pressures, _, _, _ = ddm_w.generate(
            w_init=w_inits[idx:idx+1].to(device),
            params=params_norm[idx:idx+1].to(device),
            device=device,
        )

    wing = gen_coords[0].cpu().numpy()
    pres = gen_pressures[0].cpu().numpy()

    fig = plt.figure(figsize=(9, 5))
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    norm = draw_smooth_surface(ax, wing, pres)

    ax.set_box_aspect([1, 2.505, 0.18])
    ax.view_init(elev=22, azim=-55)

    # Remove all axis decorations
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_axis_off()

    out_png = os.path.join(save_dir, "ddmw_3d_subsonic_presentation.png")
    out_pdf = os.path.join(save_dir, "ddmw_3d_subsonic_presentation.pdf")
    for path in (out_png, out_pdf):
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {path}")

    plt.close(fig)

    from PIL import Image, ImageChops
    img  = Image.open(out_png).convert("RGB")
    bg   = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        pad  = 12
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(img.width,  bbox[2] + pad),
                min(img.height, bbox[3] + pad))
        img.crop(bbox).save(out_png)
        print(f"Cropped: {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
