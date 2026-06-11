"""
Side-by-side 3D wing comparison: DDM_3D vs DDM_W_3D vs Ground Truth.

For a selection of test cases (one per Mach regime by default), generates wings
from both models under identical conditions and plots them in a single figure.

Layout per wing (one column per test case):
  Row 0 — DDM_W   (steelblue)
  Row 1 — DDM_3D  (darkorange)
  Row 2 — GT      (dimgray)

Usage
-----
    python -m engiopt.analysis.compare_ddm_3d_vs_ddm_w \
        --ddm_w_checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
        --ddm_3d_checkpoint     results/ddm_3d/ddm_3d_run039_v1_best.pth \
        --bae_checkpoint        results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint       results/lvae_3d/lvae_3d_v29_best.pth \
        --out                   results/plots/compare_ddm_3d_vs_ddm_w.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.ddm.train_ddm_3d import DDM3D, load_bae_3d, precompute_z
from engiopt.ddm.evaluate_ddm_3d import generate as generate_ddm_3d
from engiopt.ddm import samplers
from engiopt.ddm.plotting import wing_3D_shape_plot
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def load_ddm_w(args, bae_model, device):
    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    lvae_model = load_lvae_3d(cfg, bae_model)
    ckpt = torch.load(args.ddm_w_checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)

    w_dim = cfg.lae_latent_dim
    saved = ckpt["denoiser"]
    denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
    if not isinstance(saved, MLPDenoiser):
        denoiser.load_state_dict(saved)
    else:
        denoiser = saved

    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")

    model = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
        name="ddm_w",
    )
    model.w_mean     = ckpt.get("w_mean")
    model.w_std      = ckpt.get("w_std")
    model.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    model.p_ddm_std  = ckpt.get("p_ddm_std", 1.0)
    model.denoiser.to(device)
    print("DDM_W loaded.")
    return model, lvae_model, pms, ams


def load_ddm_3d(args, bae_model, device):
    ckpt  = torch.load(args.ddm_3d_checkpoint, map_location="cpu", weights_only=False)
    z_dim = ckpt.get("z_dim", 64)
    pms   = ckpt.get("params_mean_std")
    ams   = ckpt.get("aoas_mean_std")

    sampler  = samplers.BaselineSampler_AoA_3D(1000, start_x=1e-4, end_x=0.02,
                                                start_alpha=1e-4, end_alpha=0.02)
    net_keys = [k for k in ckpt["denoiser"] if k.startswith("net.") and k.endswith(".weight")]
    spacing  = int(net_keys[1].split(".")[1]) - int(net_keys[0].split(".")[1])
    dropout  = 0.1 if spacing == 3 else 0.0
    denoiser = MLPDenoiser(w_dim=z_dim, c_dim=4, dropout=dropout)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.to(device).eval()

    model = DDM3D(
        denoiser=denoiser, bae_model=bae_model, sampler=sampler,
        z_dim=z_dim, c_dim=4, params_mean_std=pms, aoas_mean_std=ams,
        name="ddm_3d",
    )
    model.z_mean = ckpt.get("z_mean")
    model.z_std  = ckpt.get("z_std")
    print("DDM_3D loaded.")
    return model, pms, ams


def pick_cases(test_dataset, n_per_regime=1, seed=0):
    """Pick n_per_regime cases per Mach regime, spread across the test set."""
    rng = np.random.default_rng(seed)
    regimes = {
        "subsonic":   [i for i, it in enumerate(test_dataset) if it["mach"] < 0.8],
        "transonic":  [i for i, it in enumerate(test_dataset) if 0.8 <= it["mach"] < 1.0],
        "supersonic": [i for i, it in enumerate(test_dataset) if it["mach"] >= 1.0],
    }
    idxs = []
    for name, pool in regimes.items():
        if pool:
            chosen = rng.choice(pool, min(n_per_regime, len(pool)), replace=False)
            idxs.extend(chosen.tolist())
            print(f"  {name}: cases {[test_dataset[i]['case_num'] for i in chosen]}")
    return sorted(idxs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ddm_w_checkpoint",  type=str, required=True)
    p.add_argument("--ddm_3d_checkpoint", type=str, required=True)
    p.add_argument("--bae_checkpoint",    type=str, required=True)
    p.add_argument("--lvae_checkpoint",   type=str, required=True)
    p.add_argument("--n_per_regime", type=int, default=1)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--out",          type=str, default="results/plots/compare_ddm_3d_vs_ddm_w.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    # ── Load BAE (shared) ─────────────────────────────────────────────────────
    bae_model = load_bae_3d(args.bae_checkpoint, device, n_spans=15)

    # ── Load both models ──────────────────────────────────────────────────────
    ddm_w,  lvae_model, pms_w, ams_w = load_ddm_w(args, bae_model, device)
    ddm_3d, pms_3d, ams_3d           = load_ddm_3d(args, bae_model, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test = list(dataset["test"])
    initial_by_case = {it["case_num"]: it for it in all_test if it["initial"] == 1}
    test_finals     = [it for it in all_test if it["final"] == 1]

    print("Picking test cases:")
    idxs = pick_cases(test_finals, n_per_regime=args.n_per_regime, seed=args.seed)
    selected = [test_finals[i] for i in idxs]
    n_wings  = len(selected)

    # ── Precompute for DDM_W ──────────────────────────────────────────────────
    scaler_params_w = scaler(pms_w) if pms_w is not None else None
    scaler_aoas_w   = scaler(ams_w) if ams_w is not None else None

    gt_coords_w, gt_aoas_w, _, _, _, w_inits, params_norm_w, case_nums = \
        precompute_test_3d(
            selected, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std,
            scaler_params_w, scaler_aoas_w, device,
        )

    # ── Precompute for DDM_3D ─────────────────────────────────────────────────
    z_gt, aoas_gt_3d, params_all_3d, z_inits_3d = precompute_z(
        selected, initial_by_case, bae_model, device
    )
    gt_coords_3d = bae_model.decode(z_gt.to(device)).cpu()

    scaler_params_3d = scaler(pms_3d) if pms_3d is not None else None
    z_mean = ddm_3d.z_mean
    z_std  = ddm_3d.z_std
    z_inits_n  = (z_inits_3d - z_mean) / z_std if z_mean is not None else z_inits_3d
    params_n_3d = torch.stack([
        scaler_params_3d.transform(p) for p in params_all_3d
    ]) if scaler_params_3d is not None else params_all_3d

    # ── Generate ─────────────────────────────────────────────────────────────
    print("Generating from DDM_W...")
    gen_coords_w, gen_aoas_w, _, _, _, _ = ddm_w.generate(
        w_init=w_inits.to(device),
        params=params_norm_w.to(device),
        device=device,
    )

    print("Generating from DDM_3D...")
    gen_coords_3d, gen_aoas_3d, _ = generate_ddm_3d(
        ddm_3d, z_inits_n, params_n_3d, device
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    row_labels  = ["DDM_W (LVAE)", "DDM_3D (BAE)", "Ground Truth"]
    row_colors  = ["steelblue",    "darkorange",    "dimgray"]
    row_coords  = [gen_coords_w,   gen_coords_3d,   gt_coords_w]
    row_aoas    = [gen_aoas_w,     gen_aoas_3d,     gt_aoas_w]
    n_rows = len(row_labels)

    wing_len = 2.25
    fig = plt.figure(figsize=(5.5 * n_wings, 5.0 * n_rows))

    for col, item in enumerate(selected):
        mach = item["mach"]
        regime = "Subsonic" if mach < 0.8 else ("Transonic" if mach < 1.0 else "Supersonic")
        cond_str = (f"{regime}  M={mach:.2f}\n"
                    f"Re={item['reynolds']/1e6:.1f}M  CL={item['cl_target']:.2f}")

        for row in range(n_rows):
            ax = fig.add_subplot(n_rows, n_wings, row * n_wings + col + 1, projection='3d')
            coords = row_coords[row][col].numpy()   # [S, 2, 192]
            aoa    = float(np.atleast_1d(np.array(row_aoas[row]))[col])
            z_pos  = torch.linspace(0, 1, coords.shape[0]) * wing_len

            wing_3D_shape_plot(coords, ax=ax, facecolor=row_colors[row],
                               alpha=0.75, z=z_pos, wing_len=wing_len)
            ax.set_axis_off()

            title = f"{row_labels[row]}\nAoA={aoa:.1f}°"
            if row == 0:
                title = f"Case {item['case_num']}  {cond_str}\n{title}"
            ax.set_title(title, fontsize=8, pad=2)

    fig.suptitle(
        "DDM_W (LVAE latent, 20D)  vs  DDM_3D (BAE latent, 128D)  vs  Ground Truth",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
