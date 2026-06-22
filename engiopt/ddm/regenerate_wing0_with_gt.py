"""Re-run the 2D-DDM generation pipeline for a single test wing to recover the
ground-truth pairing (coords + flow conditions) that was not persisted by the
original generate_airfoils.py run, alongside the matching generated sample.

Usage:
    python -m engiopt.ddm.regenerate_wing0_with_gt [wing_index]
"""
import os
import sys

import numpy as np
import torch

from engibench.problems.wings3D.v0 import Wings3D

from engiopt.ddm.train_ddm import Config, load_bae, build_unet, build_sampler
from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.generate_airfoils import generate_from_test_items

OUT_DIR = "results/generated/20260401_153629_v7hope250:1"


def main():
    wing_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_path = os.path.join(OUT_DIR, f"wing{wing_idx:02d}_with_gt.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = Config()
    cfg.model_name = "ddm_v7hope(250:1)"

    bae_model = load_bae(cfg)
    unet = build_unet(cfg)
    sampler = build_sampler(cfg)

    ddm_model = DDM_AoAInit_3D(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        params_mean_std=(0, 1),
        aoas_mean_std=(0, 1),
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )
    checkpoint_path = f"{cfg.save_dir}/{cfg.model_name}.pth"
    print(f"Loading checkpoint from {checkpoint_path}...")
    ddm_model.load(checkpoint_path, train_mode=False)
    ddm_model.unet = ddm_model.unet.to(device)
    ddm_model.bae_model = ddm_model.bae_model.to(device)

    problem_gt = Wings3D(seed=cfg.seed)
    all_test = list(problem_gt.dataset["test"])
    gt_finals = [item for item in all_test if item["final"] == 1]
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}

    test_items_subset = gt_finals[wing_idx:wing_idx + 1]
    generated_airfoils, generated_alphas, _, paired_gt_coords, paired_gt_alphas = \
        generate_from_test_items(ddm_model, bae_model, test_items_subset, initial_by_case, device)

    gt_item = test_items_subset[0]
    torch.save({
        "generated_airfoil": generated_airfoils[0].cpu(),
        "generated_alpha": float(generated_alphas[0]),
        "gt_coords": torch.as_tensor(np.array(paired_gt_coords[0])),  # [9, 192, 2]
        "gt_alpha": float(paired_gt_alphas[0]),
        "mach": float(gt_item["mach"]),
        "reynolds": float(gt_item["reynolds"]),
        "cl_target": float(gt_item["cl_target"]),
    }, out_path)
    print(f"Saved {out_path}")
    print(f"Generated alpha: {float(generated_alphas[0]):.2f}  |  GT alpha: {paired_gt_alphas[0]:.2f}")
    print(f"Mach={gt_item['mach']:.2f}  Re={gt_item['reynolds']:.2e}  CL={gt_item['cl_target']:.2f}")


if __name__ == "__main__":
    main()
