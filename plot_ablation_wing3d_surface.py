"""
4-panel thesis figure: DDM_W3D ablation wing surfaces in the same style as
figure 22a (bae3d_gt_vs_recon). One generated wing per selected n, rendered
as a filled surface coloured by y/c, arranged in a single row.

Usage:
    python plot_ablation_wing3d_surface.py \
        [--ablation_dir results/ddm_w_3d_ablation_v29] \
        [--bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt] \
        [--lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth] \
        [--wing_idx 0] \
        [--out results/ablation_wing3d_surface.pdf]
"""

import argparse
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

from engiopt.lvae.evaluate_lvae_3d import load_bae_3d, load_lvae_3d
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d as load_lvae_ddm, build_sampler
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
_SPAN_CHORD_RATIO = 2.505

SELECTED = [50, 100, 300, 400, 767]

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi":     150,
    "text.usetex":    False,
})


def _draw_surface(ax, data: np.ndarray):
    """Draw thesis-style surface mesh. data: [S, 2, N]."""
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

    ax.set_xlabel("x/c", labelpad=4)
    ax.set_ylabel("Span η", labelpad=14)
    ax.set_zlabel("y/c", labelpad=4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect([1, _SPAN_CHORD_RATIO, 1])
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.25))
    return surf


def _draw_cp_surface(ax, wing: np.ndarray, pres: np.ndarray,
                     pmin: float, pmax: float, n_pts: int = 128):
    """Cp-coloured upper/lower surface (same style as plot_thesis_ddmw_3d)."""
    S, _, N = wing.shape
    cmap = _cm.get_cmap("RdBu_r")
    norm = _mcolors.Normalize(vmin=pmin, vmax=pmax)
    span_pos = np.linspace(0.0, _SPAN_CHORD_RATIO, S)

    upper_x = np.zeros((S, n_pts)); upper_y = np.zeros((S, n_pts)); upper_p = np.zeros((S, n_pts))
    lower_x = np.zeros((S, n_pts)); lower_y = np.zeros((S, n_pts)); lower_p = np.zeros((S, n_pts))

    for s in range(S):
        x, y, p = wing[s, 0], wing[s, 1], pres[s]
        le = int(np.argmin(x))
        for idx, ux, uy, up in [(np.arange(0, le + 1), upper_x, upper_y, upper_p),
                                  (np.arange(le, N),    lower_x, lower_y, lower_p)]:
            t0 = np.linspace(0, 1, len(idx))
            t1 = np.linspace(0, 1, n_pts)
            ux[s] = interp1d(t0, x[idx], kind="linear")(t1)
            uy[s] = interp1d(t0, y[idx], kind="linear")(t1)
            up[s] = interp1d(t0, p[idx], kind="linear")(t1)

    span_grid = np.tile(span_pos[:, None], (1, n_pts))
    for X, Z, P in [(upper_x, upper_y, upper_p), (lower_x, lower_y, lower_p)]:
        ax.plot_surface(X, span_grid, Z, facecolors=cmap(norm(P)),
                        alpha=1.0, linewidth=0, antialiased=True,
                        rcount=S, ccount=n_pts)

    ax.set_xlabel("x/c", labelpad=4)
    ax.set_ylabel("Span η", labelpad=14)
    ax.set_zlabel("")
    ax.set_box_aspect([1, _SPAN_CHORD_RATIO, 0.18])
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.zaxis.set_major_locator(plt.NullLocator())


def load_ddm_w_checkpoint(ckpt_path, bae_model, lvae_model, cfg, device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)
    w_dim   = cfg.lae_latent_dim
    saved   = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)
    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")
    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(ckpt_path))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    return ddm_w, pms, ams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir",   default="results/ddm_w_3d_ablation_v29")
    p.add_argument("--bae_checkpoint", default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--lvae_checkpoint", default="results/lvae_3d/lvae_3d_v29_best.pth")
    p.add_argument("--wing_idx", type=int, default=0, help="Which test wing to render")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out", default="results/ablation_wing3d_surface.pdf")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    bae_model = load_bae_3d(args.bae_checkpoint, device)

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device          = device
    lvae_model = load_lvae_ddm(cfg, bae_model)

    # Precompute test set once (shared across all checkpoints)
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings, rendering wing_idx={args.wing_idx}")

    # Load one checkpoint just to get w_mean/w_std for precompute (use n767 as reference)
    ref_ckpt_path = os.path.join(args.ablation_dir, "n767_s0",
                                 "ddm_w_3d_ablation_n767_s0_best.pth")
    ref_ckpt = torch.load(ref_ckpt_path, map_location="cpu", weights_only=False)
    pms_ref  = ref_ckpt.get("params_mean_std")
    ams_ref  = ref_ckpt.get("aoas_mean_std")
    scaler_params = scaler(pms_ref) if pms_ref is not None else None
    scaler_aoas   = scaler(ams_ref) if ams_ref is not None else None

    gt_coords, gt_aoas, gt_pressures, gt_w, gt_z, w_inits, params_norm, case_nums = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            ref_ckpt.get("w_mean"), ref_ckpt.get("w_std"),
            scaler_params, scaler_aoas, device,
        )

    idx = args.wing_idx
    w_init_i  = w_inits[idx:idx+1].to(device)
    params_i  = params_norm[idx:idx+1].to(device)

    gt_wing     = gt_coords[idx].cpu().numpy()    # [S, 2, 192]
    gt_pres_raw = gt_pressures[idx].cpu().numpy() # [S, 192]

    # Generate one wing + pressure per selected n
    gen_wings = {}
    gen_pres  = {}
    for n in SELECTED:
        ckpt_path = os.path.join(args.ablation_dir, f"n{n}_s0",
                                 f"ddm_w_3d_ablation_n{n}_s0_best.pth")
        print(f"Loading n={n}: {ckpt_path}")
        ddm_w, pms, ams = load_ddm_w_checkpoint(ckpt_path, bae_model, lvae_model, cfg, device)
        with torch.no_grad():
            coords, _, pres_t, _, _, _ = ddm_w.generate(
                w_init=w_init_i, params=params_i, device=device,
            )
        gen_wings[n] = coords.squeeze(0).cpu().numpy()
        gen_pres[n]  = pres_t.squeeze(0).cpu().numpy()
        print(f"  Generated wing shape: {gen_wings[n].shape}")

    n_gen = len(SELECTED)
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    all_pres = np.concatenate([gt_pres_raw.ravel()] + [p.ravel() for p in gen_pres.values()])
    pmin = float(np.nanpercentile(all_pres, 2))
    pmax = float(np.nanpercentile(all_pres, 98))

    # ── Combined 3-row figure ─────────────────────────────────────────────────
    # Row 0: GT only (geometry left, Cp right, centred)
    # Row 1: generated geometry, n_gen panels
    # Row 2: generated Cp,       n_gen panels
    fig = plt.figure(figsize=(4.5 * n_gen, 13))

    # GT row: two panels side by side, centred via gridspec
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs_top = GridSpec(1, n_gen, figure=fig,
                      left=0.02, right=0.88, top=0.97, bottom=0.68,
                      wspace=0.05)
    gt_geom_col = (n_gen - 2) // 2
    ax_gt_geom = fig.add_subplot(gs_top[0, gt_geom_col], projection="3d")
    _draw_surface(ax_gt_geom, gt_wing)
    ax_gt_geom.set_title("Ground truth — geometry", pad=8)

    ax_gt_cp = fig.add_subplot(gs_top[0, gt_geom_col + 1], projection="3d")
    _draw_cp_surface(ax_gt_cp, gt_wing, gt_pres_raw, pmin, pmax)
    ax_gt_cp.set_title("Ground truth — $C_p$", pad=8)
    ax_gt_cp.set_xlabel(""); ax_gt_cp.set_ylabel(""); ax_gt_cp.set_zlabel("")

    # Geom row
    gs_geom = GridSpec(1, n_gen, figure=fig,
                       left=0.02, right=0.88, top=0.64, bottom=0.35,
                       wspace=0.05)
    for col, n in enumerate(SELECTED):
        ax = fig.add_subplot(gs_geom[0, col], projection="3d")
        _draw_surface(ax, gen_wings[n])
        ax.set_title(f"n = {n}", pad=8)
        if col > 0:
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # Cp row
    gs_cp = GridSpec(1, n_gen, figure=fig,
                     left=0.02, right=0.88, top=0.31, bottom=0.02,
                     wspace=0.05)
    for col, n in enumerate(SELECTED):
        ax = fig.add_subplot(gs_cp[0, col], projection="3d")
        _draw_cp_surface(ax, gen_wings[n], gen_pres[n], pmin, pmax)
        if col == 0:
            ax.set_title(f"n = {n}", pad=8)
        else:
            ax.set_title(f"n = {n}", pad=8)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # Colorbars on the right
    cbar_ax_geom = fig.add_axes([0.90, 0.35, 0.015, 0.30])
    sm_geom = plt.cm.ScalarMappable(cmap="coolwarm")
    sm_geom.set_array([])
    fig.colorbar(sm_geom, cax=cbar_ax_geom).set_label("y/c  (surface height)", fontsize=9)

    cbar_ax_cp = fig.add_axes([0.90, 0.02, 0.015, 0.27])
    sm_cp = _cm.ScalarMappable(cmap="RdBu_r", norm=_mcolors.Normalize(vmin=pmin, vmax=pmax))
    sm_cp.set_array([])
    fig.colorbar(sm_cp, cax=cbar_ax_cp).set_label("$C_p$", fontsize=10)

    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Saved: {stem}.{ext}")
    plt.close()

    # ── GT-only figure (geom left, Cp right) ──────────────────────────────────
    fig_gt = plt.figure(figsize=(8, 4))
    ax_gt_geom = fig_gt.add_subplot(1, 2, 1, projection="3d")
    _draw_surface(ax_gt_geom, gt_wing)
    ax_gt_geom.set_zlabel("y/c", labelpad=24)
    ax_gt_cp = fig_gt.add_subplot(1, 2, 2, projection="3d")
    _draw_cp_surface(ax_gt_cp, gt_wing, gt_pres_raw, pmin, pmax)
    ax_gt_cp.set_xlabel(""); ax_gt_cp.set_ylabel(""); ax_gt_cp.set_zlabel("")
    fig_gt.subplots_adjust(right=0.86, wspace=0.4)
    cbar_gt_g = fig_gt.add_axes([0.49, 0.15, 0.012, 0.7])
    sm_gt_g = plt.cm.ScalarMappable(cmap="coolwarm"); sm_gt_g.set_array([])
    fig_gt.colorbar(sm_gt_g, cax=cbar_gt_g).set_label("y/c", fontsize=9)
    cbar_gt_c = fig_gt.add_axes([0.88, 0.15, 0.012, 0.7])
    sm_gt_c = _cm.ScalarMappable(cmap="RdBu_r", norm=_mcolors.Normalize(vmin=pmin, vmax=pmax))
    sm_gt_c.set_array([])
    fig_gt.colorbar(sm_gt_c, cax=cbar_gt_c).set_label("$C_p$", fontsize=9)
    fig_gt.savefig(f"{stem}_gt.pdf", dpi=200, bbox_inches="tight")
    print(f"Saved: {stem}_gt.pdf")
    plt.close()

    # ── Geom-only figure (5 generated panels, no GT) ──────────────────────────
    fig_g = plt.figure(figsize=(5.5 * n_gen, 5.5))
    for col, n in enumerate(SELECTED):
        ax = fig_g.add_subplot(1, n_gen, col + 1, projection="3d")
        _draw_surface(ax, gen_wings[n])
        ax.set_title(f"n = {n}", pad=8)
        if col > 0:
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    fig_g.subplots_adjust(right=0.88, wspace=0.05)
    cbar_g = fig_g.add_axes([0.90, 0.15, 0.015, 0.7])
    sm_g = plt.cm.ScalarMappable(cmap="coolwarm"); sm_g.set_array([])
    fig_g.colorbar(sm_g, cax=cbar_g).set_label("y/c  (surface height)", fontsize=9)
    fig_g.savefig(f"{stem}_geom.pdf", dpi=200, bbox_inches="tight")
    print(f"Saved: {stem}_geom.pdf")
    plt.close()

    # ── Cp-only figure (5 generated panels, no GT) ────────────────────────────
    fig_c = plt.figure(figsize=(5.5 * n_gen, 5.5))
    for col, n in enumerate(SELECTED):
        ax = fig_c.add_subplot(1, n_gen, col + 1, projection="3d")
        _draw_cp_surface(ax, gen_wings[n], gen_pres[n], pmin, pmax)
        ax.set_title(f"n = {n}", pad=8)
        if col > 0:
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    fig_c.subplots_adjust(right=0.88, wspace=0.05)
    cbar_c = fig_c.add_axes([0.90, 0.15, 0.015, 0.7])
    sm_c = _cm.ScalarMappable(cmap="RdBu_r", norm=_mcolors.Normalize(vmin=pmin, vmax=pmax))
    sm_c.set_array([])
    fig_c.colorbar(sm_c, cax=cbar_c).set_label("$C_p$", fontsize=10)
    fig_c.savefig(f"{stem}_cp.pdf", dpi=200, bbox_inches="tight")
    print(f"Saved: {stem}_cp.pdf")
    plt.close()


if __name__ == "__main__":
    main()
