"""
Plot generated vs ground-truth 3D wings for a DDM_PCA_3D checkpoint.

Layout: 2 rows × n_wings columns
  Row 0 — Generated  (steelblue)
  Row 1 — Ground Truth (coral)

Usage
-----
    python -m engiopt.ddm.ddm_pca.plot_wings_pca_3d \
        --checkpoint     results/ddm_pca_3d/ddm_pca_3d_v2_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
        [--n_wings 5] [--seed 0] [--out results/plots/wings_ddm_pca_3d.png]
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_pca.ddm_pca_3d import DDM_PCA_3D, MLPDenoiser
from engiopt.ddm.ddm_pca.train_ddm_pca_3d import Config, build_sampler
from engiopt.ddm.ddm_pca.evaluate_ddm_pca_3d import precompute_test_3d
from engiopt.ddm.plotting import wing_3D_shape_plot
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--bae_checkpoint", type=str, required=True)
    p.add_argument("--n_wings", type=int, default=5)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--T",       type=int, default=None)
    p.add_argument("--out",     type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    bae_model = load_bae_3d(args.bae_checkpoint, device)

    ckpt     = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pca_path = args.checkpoint.replace("_best.pth", "_pca.pkl").replace(".pth", "_pca.pkl")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    cfg   = Config()
    z_dim = ckpt.get("z_dim", cfg.n_components)
    pms   = ckpt.get("params_mean_std")
    ams   = ckpt.get("aoas_mean_std")

    denoiser = MLPDenoiser(z_dim=z_dim, c_dim=cfg.c_dim).to(device)
    model = DDM_PCA_3D(
        denoiser=denoiser, pca=pca, bae_model=bae_model,
        sampler=build_sampler(cfg), z_dim=z_dim, c_dim=cfg.c_dim,
        w_aoa=ckpt.get("w_aoa", 1.0),
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    model.load(args.checkpoint, train_mode=False)
    model.denoiser.to(device)
    print("Model loaded.")

    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]

    gt_coords, gt_aoas, gt_z, z_inits, params_norm = precompute_test_3d(
        test_dataset, initial_by_case, bae_model, pca,
        model.z_mean, model.z_std,
        model.scaler_params, model.scaler_aoas, device,
    )

    n_wings = min(args.n_wings, gt_coords.shape[0])
    print(f"Test set: {gt_coords.shape[0]} wings, plotting {n_wings}")

    gen_coords, gen_aoas, _ = model.generate(
        z_init=z_inits[:n_wings].to(device),
        params=params_norm[:n_wings].to(device),
        device=device, T=args.T,
    )

    wing_len = 2.25
    fig = plt.figure(figsize=(5.5 * n_wings, 10))

    for i in range(n_wings):
        item     = test_dataset[i]
        mach     = item["mach"]
        regime   = "Subsonic" if mach < 0.8 else ("Transonic" if mach < 1.0 else "Supersonic")
        cond_str = f"{regime}  M={mach:.2f}\nRe={item['reynolds']/1e6:.1f}M  CL={item['cl_target']:.2f}"
        z_pos    = torch.linspace(0, 1, gen_coords.shape[1]) * wing_len

        # Generated
        ax = fig.add_subplot(2, n_wings, i + 1, projection='3d')
        wing_3D_shape_plot(gen_coords[i].numpy(), ax=ax, facecolor='steelblue',
                           alpha=0.75, z=z_pos, wing_len=wing_len)
        ax.set_axis_off()
        gen_aoa = float(np.atleast_1d(gen_aoas.numpy())[i])
        ax.set_title(f"Case {item['case_num']}  {cond_str}\nGenerated  AoA={gen_aoa:.1f}°",
                     fontsize=8, pad=2)

        # Ground truth
        ax = fig.add_subplot(2, n_wings, n_wings + i + 1, projection='3d')
        wing_3D_shape_plot(gt_coords[i].numpy(), ax=ax, facecolor='coral',
                           alpha=0.75, z=z_pos, wing_len=wing_len)
        ax.set_axis_off()
        ax.set_title(f"Ground Truth  AoA={gt_aoas[i].item():.1f}°", fontsize=8, pad=2)

    fig.suptitle("DDM_PCA_3D — Generated (steelblue) vs Ground Truth (coral)",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    if args.out is None:
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.out = f"results/plots/wings_{model_name}.png"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
