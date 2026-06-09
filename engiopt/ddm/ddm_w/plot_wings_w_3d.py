"""
Plot generated vs ground-truth wings for a DDM_W3D checkpoint.

Usage
-----
    python -m engiopt.ddm.ddm_w.plot_wings_w_3d \
        --checkpoint      results/ddm_w_3d/ddm_w_3d_v1_best.pth \
        --bae_checkpoint  results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v5_best.pth \
        [--n_wings 6] [--seed 0] [--T 1000]
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
from engiopt.ddm.plotting import wing_3D_shape_plot, wing_3D_pressure_plot
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--n_wings", type=int, default=6)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--T",       type=int, default=None)
    p.add_argument("--out",     type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.device = device

    bae_model  = load_bae_3d(args.bae_checkpoint, device)
    lvae_model = load_lvae_3d(cfg, bae_model)

    ckpt    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)

    w_dim = cfg.lae_latent_dim
    saved = ckpt["denoiser"]
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
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    print("Model loaded.")

    w_mean = ddm_w.w_mean
    w_std  = ddm_w.w_std

    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]

    gt_coords, gt_aoas, gt_pressures, _, w_inits, params_norm, case_nums = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            w_mean, w_std, scaler_params, scaler_aoas, device,
        )

    n_test  = gt_coords.shape[0]
    n_wings = min(args.n_wings, n_test)
    print(f"Test set: {n_test} wings, plotting {n_wings}")

    gen_coords, gen_aoas, gen_pressures, _, _ = ddm_w.generate(
        w_init=w_inits[:n_wings].to(device),
        params=params_norm[:n_wings].to(device),
        device=device,
        T=args.T,
    )

    raw_flow = [
        (item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"])
        for item in test_dataset[:n_wings]
    ]

    all_p = np.concatenate([
        gt_pressures[:n_wings].numpy().ravel(),
        gen_pressures.numpy().ravel(),
    ])
    pmin = float(np.nanpercentile(all_p, 2))
    pmax = float(np.nanpercentile(all_p, 98))

    wing_len = 2.25
    n_rows   = 4
    fig = plt.figure(figsize=(10 * n_wings, 10 * n_rows), dpi=150)

    S = gen_coords.shape[1]
    etas = [torch.linspace(0, 1, S) for _ in range(n_wings)]

    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        cond_str = f"M={mach:.2f}  Re={re/1e6:.2f}M\nCL={cl:.2f}  AR={ar:.2f}"
        aoa_str  = f"{float(np.atleast_1d(gen_aoas.numpy())[i]):.1f}°"
        z_pos    = etas[i] * wing_len

        ax = fig.add_subplot(n_rows, n_wings, i + 1, projection='3d')
        wing_3D_shape_plot(gen_coords[i].numpy(), ax=ax, facecolor='steelblue', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen {i+1}  AoA={aoa_str}\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, n_wings + i + 1, projection='3d')
        wing_3D_shape_plot(gt_coords[i].numpy(), ax=ax, facecolor='coral', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT {i+1}  AoA={gt_aoas[i].item():.1f}°\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, 2 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gen_coords[i].numpy(), gen_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen Cp {i+1}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, 3 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gt_coords[i].numpy(), gt_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT Cp {i+1}", fontsize=7)
        ax.set_axis_off()

    fig.suptitle(
        f"{os.path.basename(args.checkpoint)}  —  rows: gen shape | GT shape | gen Cp | GT Cp",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    if args.out is None:
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.out = f"results/plots/wings_{model_name}.png"

    base, ext = os.path.splitext(args.out)
    out_path  = args.out
    counter   = 1
    while os.path.exists(out_path):
        out_path = f"{base}_{counter}{ext}"
        counter += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Spanwise slice plots
    slices_dir = os.path.join(os.path.dirname(out_path), "spanwise_slices_w_3d")
    os.makedirs(slices_dir, exist_ok=True)
    n_slices  = gen_coords.shape[1]
    n_cols    = min(n_slices, 5)
    n_rows_sp = (n_slices + n_cols - 1) // n_cols

    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        gen_np = gen_coords[i].numpy()
        gt_np  = gt_coords[i].numpy()
        case_num = case_nums[i]

        crossing_slices = []
        for s in range(gen_np.shape[0]):
            n_pts  = gen_np.shape[2]
            le_idx = int(np.argmin(gen_np[s, 0]))
            te_idx = int(np.argmax(gen_np[s, 0]))
            if le_idx < te_idx:
                upper_idx = np.arange(le_idx, te_idx + 1)
                lower_idx = np.concatenate([np.arange(te_idx, n_pts), np.arange(0, le_idx + 1)])
            else:
                upper_idx = np.concatenate([np.arange(le_idx, n_pts), np.arange(0, te_idx + 1)])
                lower_idx = np.arange(te_idx, le_idx + 1)
            if gen_np[s, 1, upper_idx].min() < gen_np[s, 1, lower_idx].max():
                crossing_slices.append(s)
        if crossing_slices:
            print(f"  Wing {i+1} (case {case_num}): self-intersecting slices {crossing_slices}")

        fig_sp, axes = plt.subplots(
            n_rows_sp * 2, n_cols,
            figsize=(4 * n_cols, 3 * n_rows_sp * 2),
        )
        axes = np.array(axes).reshape(n_rows_sp * 2, n_cols)

        for s in range(n_slices):
            row_gen = (s // n_cols) * 2
            row_gt  = row_gen + 1
            col     = s % n_cols
            n_pts   = gen_np.shape[2]

            for ax, arr, color, label in [
                (axes[row_gen, col], gen_np[s], 'steelblue', f"Gen  slice {s}"),
                (axes[row_gt,  col], gt_np[s],  'coral',     f"GT   slice {s}"),
            ]:
                le_idx = int(np.argmin(arr[0]))
                te_idx = int(np.argmax(arr[0]))
                if le_idx < te_idx:
                    upper = np.arange(le_idx, te_idx + 1)
                    lower = np.concatenate([np.arange(te_idx, n_pts), np.arange(0, le_idx + 1)])
                else:
                    upper = np.concatenate([np.arange(le_idx, n_pts), np.arange(0, te_idx + 1)])
                    lower = np.arange(te_idx, le_idx + 1)
                ax.plot(arr[0, upper], arr[1, upper], color=color, lw=1.5)
                ax.plot(arr[0, lower], arr[1, lower], color=color, lw=1.5)
                ax.set_title(label, fontsize=7)
                ax.set_aspect('equal')
                x_pad = (arr[0].max() - arr[0].min()) * 0.05 + 0.01
                y_pad = (arr[1].max() - arr[1].min()) * 0.10 + 0.01
                ax.set_xlim(arr[0].min() - x_pad, arr[0].max() + x_pad)
                ax.set_ylim(arr[1].min() - y_pad, arr[1].max() + y_pad)
                ax.grid(True, lw=0.4)
                ax.tick_params(labelsize=6)

        for s in range(n_slices, n_rows_sp * n_cols):
            axes[(s // n_cols) * 2,     s % n_cols].set_visible(False)
            axes[(s // n_cols) * 2 + 1, s % n_cols].set_visible(False)

        aoa_str = f"{float(np.atleast_1d(gen_aoas.numpy())[i]):.1f}"
        fig_sp.suptitle(
            f"Wing {i+1} (DDM_W3D)  |  M={mach:.2f}  Re={re/1e6:.2f}M  "
            f"CL={cl:.2f}  AR={ar:.2f}  |  gen AoA={aoa_str}°  GT AoA={gt_aoas[i].item():.1f}°\n"
            f"Blue = generated,  Coral = ground truth",
            fontsize=9,
        )
        fig_sp.tight_layout()
        slice_path = os.path.join(slices_dir, f"spanwise_wing_{i+1:02d}.png")
        fig_sp.savefig(slice_path, bbox_inches='tight', dpi=150)
        plt.close(fig_sp)
        print(f"Saved spanwise slices to {slice_path}")


if __name__ == "__main__":
    main()
