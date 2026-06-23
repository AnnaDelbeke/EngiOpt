"""
Spanwise error and variance figures.

Usage:
    .venv/bin/python plot_spanwise.py --component variance
    .venv/bin/python plot_spanwise.py --component bae [--checkpoint ...]
    .venv/bin/python plot_spanwise.py --component ddmw \
        --checkpoint results/ddm_w_3d/ddm_w_3d_v29_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth
    .venv/bin/python plot_spanwise.py --component compare \
        --ddmw_checkpoint results/ddm_w_3d/ddm_w_3d_v29_best.pth ...
"""

import argparse
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = "thesis/figures"
os.makedirs(OUT, exist_ok=True)

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
GAMMAS       = [0.5, 25, 50, 100]
WING_LEN     = 2.55


# ── shared helpers ─────────────────────────────────────────────────────────────

def gaussian_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def mmd_per_span(gen, gt):
    S    = gen.shape[1]
    vals = []
    for s in range(S):
        g = gen[:, s].reshape(gen.shape[0], -1)
        r = gt[:, s].reshape(gt.shape[0], -1)
        n, m     = g.shape[0], r.shape[0]
        span_mmd = []
        for gamma in GAMMAS:
            Kxx = gaussian_kernel(g, g, gamma)
            Kyy = gaussian_kernel(r, r, gamma)
            Kxy = gaussian_kernel(g, r, gamma)
            span_mmd.append((Kxx.sum()/(n*n) - 2*Kxy.sum()/(n*m) + Kyy.sum()/(m*m)).item())
        vals.append(float(np.mean(span_mmd)))
    return np.array(vals)


def vendi_per_span(samples):
    S    = samples.shape[1]
    vals = []
    for s in range(S):
        x     = samples[:, s].reshape(samples.shape[0], -1)
        valid = torch.isfinite(x).all(dim=-1)
        x     = x[valid]
        if x.shape[0] < 2:
            vals.append(float("nan"))
            continue
        span_vendis = []
        for gamma in GAMMAS:
            K  = gaussian_kernel(x, x, gamma) / x.shape[0]
            K  = K + 1e-4 * torch.eye(K.shape[0], device=K.device)
            ev = torch.linalg.eigvalsh(K).clamp(min=1e-10)
            ev = ev / ev.sum()
            span_vendis.append((-(ev * ev.log()).sum()).exp().item())
        vals.append(float(np.nanmean(span_vendis)))
    return np.array(vals)


# ── raw dataset variance ───────────────────────────────────────────────────────

def plot_variance(args):
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    dataset   = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_items = list(dataset["train"]) + list(dataset["val"]) + list(dataset["test"])
    all_items = [it for it in all_items if it["final"] == 1]
    print(f"Wings: {len(all_items)}")

    coords     = np.stack([it["coords"] for it in all_items])  # [N, S, 192, 2]
    eta        = all_items[0]["transforms"]
    mean_shape = coords.mean(axis=0, keepdims=True)
    var_per_span = ((coords - mean_shape) ** 2).mean(axis=0).mean(axis=(1, 2))  # [S]

    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.plot(eta, var_per_span, color="steelblue", marker="o", ms=4, lw=1.8)
    ax.set_xlabel(r"Span position $z$ (root = 0, tip $\approx$ 2.25)", fontsize=9)
    ax.set_ylabel("Mean coordinate variance across dataset", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = args.out or f"{OUT}/raw_spanwise_variance.pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── BAE spanwise MSE ──────────────────────────────────────────────────────────

@torch.no_grad()
def plot_bae(args):
    from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
    from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    from torch.utils.data import random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_spans           = ckpt.get("n_spans", 15)
    n_control_points  = ckpt.get("n_control_points", 32)
    slice_hidden_dims = ckpt.get("slice_hidden_dims", [256, 128])
    span_hidden_dims  = ckpt.get("span_hidden_dims", [256, 128])
    latent_dim        = ckpt.get("latent_dim", 256)
    cpx_bound         = ckpt.get("cpx_bound", [0.0, 1.0])
    cpy_bound         = ckpt.get("cpy_bound", [-0.75, 0.75])

    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items   = [it for it in list(new_dataset["train"]) + list(new_dataset["val"]) if it["final"] == 1]
    full_ds     = WingsBezierDataset3D(all_items, num_extra_tip_slices=n_spans - 15)
    train_size  = int(0.9 * len(full_ds))
    _, val_ds   = random_split(full_ds, [train_size, len(full_ds) - train_size],
                               generator=torch.Generator().manual_seed(0))
    print(f"Val wings: {len(val_ds)}")

    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=n_control_points, n_data_points=192,
        slice_hidden_dims=slice_hidden_dims, span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim, cpx_bound=cpx_bound, cpy_bound=cpy_bound,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    span_mse = np.zeros(n_spans)
    span_sq  = np.zeros(n_spans)
    n_wings  = len(val_ds)
    for i in range(n_wings):
        x = val_ds[i].unsqueeze(0).to(device)
        y, _ = model(x)
        mse  = ((x - y) ** 2).mean(dim=(2, 3)).squeeze(0).cpu().numpy()
        span_mse += mse
        span_sq  += mse ** 2
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_wings}")
    mean_mse = span_mse / n_wings
    std_mse  = np.sqrt(span_sq / n_wings - mean_mse ** 2)

    fig, ax = plt.subplots(figsize=(max(8, n_spans // 2), 4))
    bars   = ax.bar(np.arange(1, n_spans + 1), mean_mse, yerr=std_mse, capsize=3,
                    color="steelblue", alpha=0.8, ecolor="gray")
    worst_k = max(1, n_spans // 5)
    for idx in np.argsort(mean_mse)[-worst_k:]:
        bars[idx].set_color("tomato")
    ax.set_xlabel("Spanwise position (1 = root, last = tip)", fontsize=11)
    ax.set_ylabel("Mean reconstruction MSE", fontsize=11)
    ax.set_xticks(np.arange(1, n_spans + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    run_dir  = args.run_dir or os.path.dirname(os.path.dirname(args.checkpoint))
    save_dir = os.path.join(run_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aggregate_spanwise_mse.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── DDM-W spanwise metrics ────────────────────────────────────────────────────

@torch.no_grad()
def _load_ddmw(checkpoint, bae_checkpoint, lvae_checkpoint, device):
    from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
    from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
    from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
    from engiopt.lvae.evaluate_lvae_3d import load_bae_3d

    cfg = Config()
    cfg.bae_checkpoint  = bae_checkpoint
    cfg.lvae_checkpoint = lvae_checkpoint
    cfg.device          = device

    bae_model  = load_bae_3d(bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    ckpt    = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)
    w_dim   = cfg.lae_latent_dim
    saved   = ckpt["denoiser"]
    from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
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
        name=os.path.splitext(os.path.basename(checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    return ddm_w, bae_model, lvae_model, pms, ams


def plot_ddmw(args):
    from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    from engiopt.data_processing.utils import scaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ddm_w, bae_model, lvae_model, pms, ams = _load_ddmw(
        args.checkpoint, args.bae_checkpoint, args.lvae_checkpoint, device
    )
    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    dataset         = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_dataset    = [it for it in all_test if it["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    gt_coords, gt_aoas, gt_pressures, gt_w, gt_z, w_inits, params_norm, _ = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std, scaler_params, scaler_aoas, device,
        )

    all_coords, all_pressures = [], []
    for i in range(args.n_passes):
        coords_p, _, pres_p, _, _, _ = ddm_w.generate(
            w_init=w_inits.to(device), params=params_norm.to(device), device=device,
        )
        all_coords.append(coords_p.cpu())
        all_pressures.append(pres_p.cpu())
        print(f"  Pass {i+1}/{args.n_passes}")

    gen_coords    = torch.stack(all_coords).mean(0)
    gen_pressures = torch.stack(all_pressures).mean(0)

    S   = gen_coords.shape[1]
    eta = np.linspace(0, WING_LEN, S)

    shape_mse_per_span   = ((gen_coords - gt_coords) ** 2).mean(dim=(0, 2, 3)).numpy()
    mmd_vals             = mmd_per_span(
        gen_coords.reshape(*gen_coords.shape[:2], -1),
        gt_coords.reshape(*gt_coords.shape[:2],  -1),
    )
    vendi_gen = vendi_per_span(gen_coords.reshape(*gen_coords.shape[:2], -1))
    vendi_gt  = vendi_per_span(gt_coords.reshape(*gt_coords.shape[:2],  -1))
    pressure_mse_per_span = ((gen_pressures - gt_pressures) ** 2).mean(dim=(0, 2)).numpy()

    BLUE  = "#4C72B0"
    GRAY  = "#888888"
    GREEN = "#55A868"
    span_labels = [str(i + 1) for i in range(S)]

    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    axes[0].plot(eta, shape_mse_per_span, color=BLUE, marker="o", ms=4, lw=1.5)
    axes[0].set_ylabel("Shape MSE", fontsize=9)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].grid(True, lw=0.4, alpha=0.6)

    axes[1].plot(eta, pressure_mse_per_span, color="darkorange", marker="o", ms=4, lw=1.5)
    axes[1].set_ylabel("Pressure MSE", fontsize=9)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].grid(True, lw=0.4, alpha=0.6)

    axes[2].plot(eta, mmd_vals, color="tab:red", marker="o", ms=4, lw=1.5)
    axes[2].set_ylabel("MMD", fontsize=9)
    axes[2].grid(True, lw=0.4, alpha=0.6)

    axes[3].plot(eta, vendi_gen, color=GREEN, marker="o", ms=4, lw=1.5, label="Generated")
    axes[3].plot(eta, vendi_gt,  color=GRAY,  marker="s", ms=4, lw=1.5, linestyle="--", label="Ground Truth")
    axes[3].set_ylabel("Vendi Score", fontsize=9)
    axes[3].legend(fontsize=8, loc="upper left")
    axes[3].grid(True, lw=0.4, alpha=0.6)

    for ax in axes:
        ax.set_xticks(eta)
        ax.set_xticklabels(span_labels, fontsize=8)
    axes[-1].set_xlabel("Span slice", fontsize=10)

    out = args.out or f"{OUT}/spanwise_metrics_ddmw.pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    if out.endswith(".pdf"):
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── baseline vs DDM-W comparison ─────────────────────────────────────────────

def plot_compare(args):
    from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
    from engiopt.ddm.ddm import DDM_AoAInit_3D
    from engiopt.ddm.train_ddm import precompute_latents, Config
    from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    from engiopt.data_processing.utils import scaler
    from engibench.problems.wings3D.v0 import Wings3D

    _V7HOPE_CKPT = "results/ddm/ddm_v7hope(250:1).pth"
    _BAE_2D_CKPT = "results/bezier_ae/run_006/models/bezier_ae_best.pt"
    BLUE   = "#4C72B0"
    ORANGE = "darkorange"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── baseline ──────────────────────────────────────────────────────────────
    cfg_2d = Config()
    cfg_2d.bae_checkpoint = _BAE_2D_CKPT
    cfg_2d.device = device
    bae_2d = BezierAutoencoder(
        n_control_points=cfg_2d.n_control_points, n_data_points=cfg_2d.n_data_points,
        batch_size=cfg_2d.bae_batch_size, auto_batch=True,
    ).to(device)
    ckpt_2d = torch.load(_BAE_2D_CKPT, map_location=device, weights_only=False)
    bae_2d.load_state_dict(ckpt_2d["model_state_dict"])
    bae_2d.eval()

    ckpt_v7 = torch.load(_V7HOPE_CKPT, map_location="cpu", weights_only=False)
    ddm_model = DDM_AoAInit_3D(
        unet=ckpt_v7["unet"].to(device),
        sampler=ckpt_v7["sampler"],
        bae_model=bae_2d,
        params_mean_std=ckpt_v7["params_mean_std"],
        aoas_mean_std=ckpt_v7["aoas_mean_std"],
    )
    ddm_model.unet.eval()
    ddm_model.latent_mean = ckpt_v7.get("latent_mean", 0.0)
    ddm_model.latent_std  = ckpt_v7.get("latent_std",  1.0)

    problem         = Wings3D(seed=args.seed)
    all_test        = list(problem.dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_final      = [it for it in all_test if it["final"] == 1]
    eta_9 = np.array(test_final[0]["transforms"], dtype=np.float32)

    z_opts, gt_aoas, params_all, z_inits = precompute_latents(test_final, initial_by_case, bae_2d, device)
    N, S = z_opts.shape[:2]
    L    = z_opts.shape[-1]

    gt_coords_list = []
    for i in range(N):
        slices = []
        for s in range(S):
            z_s = z_opts[i, s].unsqueeze(0).to(device)
            coords_s = bae_2d.decode_z(z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False)[0].squeeze(0).cpu()
            slices.append(coords_s)
        gt_coords_list.append(torch.stack(slices))
    gt_coords_2d = torch.stack(gt_coords_list)

    lat_mean = ddm_model.latent_mean
    lat_std  = ddm_model.latent_std
    if isinstance(lat_mean, torch.Tensor):
        lat_mean = lat_mean.to(device)
        lat_std  = lat_std.to(device)
        lat_mean_init = lat_mean[0]
        lat_std_init  = lat_std[0]
    else:
        lat_mean_init = lat_mean
        lat_std_init  = lat_std

    all_gen = []
    for pass_i in range(args.n_passes):
        gen_list = []
        for i in range(N):
            z_init_raw = z_inits[i].unsqueeze(0).to(device)
            z_init_b   = (z_init_raw - lat_mean_init) / lat_std_init
            params_b   = ddm_model.scaler_params.transform(params_all[i].unsqueeze(0)).to(device)
            noise_x    = torch.randn(1, S, 3, L, device=device)
            noise_alpha = torch.randn(1, 1, device=device)
            gen_z, _ = ddm_model([noise_x, noise_alpha], params_b, z_init_b, output_decoded=False)
            gen_z_raw = gen_z * lat_std + lat_mean
            gen_z_flat = gen_z_raw.squeeze(0)
            gen_z_flat[:, 0, :]  = gen_z_flat[:, 0, :].clamp(0.1, 2.0)
            gen_z_flat[:, 1:, :] = gen_z_flat[:, 1:, :].clamp(-2.228, 3.117)
            gen_af = ddm_model.bae_model.decode_z(gen_z_flat, z_ae_mode=True,
                                                    denormalize_output=False, normalized_data=False)[0]
            gen_list.append(gen_af.cpu())
        all_gen.append(torch.stack(gen_list))
        print(f"  2D pass {pass_i+1}/{args.n_passes}")
    gen_2d    = torch.stack(all_gen, dim=0).mean(0)
    mse_9     = np.nanmean(((gen_2d - gt_coords_2d) ** 2).numpy(), axis=(0, 2, 3))

    # ── DDM-W ────────────────────────────────────────────────────────────────
    ddm_w, bae_3d, lvae, pms, ams = _load_ddmw(
        args.ddmw_checkpoint, args.bae_checkpoint, args.lvae_checkpoint, device
    )
    sc_params = scaler(pms) if pms is not None else None
    sc_aoas   = scaler(ams) if ams is not None else None

    ds3d            = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test_3d     = list(ds3d["test"])
    init_by_case    = {it["case_num"]: it for it in all_test_3d if it["initial"] == 1}
    test_3d         = [it for it in all_test_3d if it["final"] == 1]
    eta_15 = np.array(test_3d[0]["transforms"], dtype=np.float32)

    gt_coords_3d, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
        test_3d, init_by_case, bae_3d, lvae,
        ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, device,
    )
    all_coords_3d = []
    for pass_i in range(args.n_passes):
        coords_p, _, _, _, _, _ = ddm_w.generate(
            w_init=w_inits.to(device), params=params_norm.to(device), device=device,
        )
        all_coords_3d.append(coords_p.cpu())
        print(f"  DDM-W pass {pass_i+1}/{args.n_passes}")
    gen_3d = torch.stack(all_coords_3d, dim=0).mean(0)
    mse_15 = ((gen_3d - gt_coords_3d) ** 2).mean(dim=(0, 2, 3)).numpy()

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(eta_9,  mse_9,  color=ORANGE, marker="o", ms=4, lw=1.5, label="2D DDM (reproduced)")
    ax.plot(eta_15, mse_15, color=BLUE,   marker="s", ms=4, lw=1.5, label="DDM-W")
    ax.set_xlabel("Spanwise position $\\eta$ (m)", fontsize=10)
    ax.set_ylabel("Shape MSE", fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.6)
    out = args.out or f"{OUT}/spanwise_mse_baseline_vs_ddmw.pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    if out.endswith(".pdf"):
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component", required=True,
                   choices=["variance", "bae", "ddmw", "compare"])
    p.add_argument("--checkpoint",      type=str,
                   default="results/bezier_ae_3d/run_041/models/bezier_ae_3d_best.pt")
    p.add_argument("--run_dir",         type=str, default=None)
    p.add_argument("--bae_checkpoint",  type=str,
                   default="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt")
    p.add_argument("--lvae_checkpoint", type=str,
                   default="results/lvae_3d/lvae_3d_v29_best.pth")
    p.add_argument("--ddmw_checkpoint", type=str,
                   default="results/ddm_w_3d/ddm_w_3d_v29_best.pth")
    p.add_argument("--n_passes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out",      type=str, default=None)
    args = p.parse_args()

    if args.component == "variance":
        plot_variance(args)
    elif args.component == "bae":
        plot_bae(args)
    elif args.component == "ddmw":
        plot_ddmw(args)
    elif args.component == "compare":
        plot_compare(args)


if __name__ == "__main__":
    main()
