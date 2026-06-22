"""
Manifold Distance scatter plot for DDM-W.

For each generated wing, compute:
  - X: distance to nearest training-set neighbour in w-space (novelty)
  - Y: Shape MSE against ground-truth optimised wing (fidelity)

Also plot the training-set self-NN distances as a gray reference cloud.

Color-code generated points by Mach regime:
  subsonic  M < 0.8   → blue
  transonic 0.8–1.0   → orange
  supersonic M >= 1.0 → red
"""

import sys, os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.lvae.train_lvae_3d import LVAE3D, LAEEncoderJoint3D
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.train_ddm_w_3d import load_bae_3d, load_lvae_3d, Config, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

# reset usetex (matplotlib.pylab sets it True on import)
plt.rcParams.update({"text.usetex": False})

# ── paths ──────────────────────────────────────────────────────────────────
BAE_CKP   = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP  = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP  = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"
OUT_PATH  = "thesis/figures/manifold_distance_scatter.png"
SLICES    = "Wing_TL/data/processed/new_dataset_slices.pkl"
SCALARS   = "Wing_TL/data/processed/new_dataset_scalars.pkl"
DEVICE    = "cpu"
N_PASSES  = 3   # average over passes


def load_ddm_w(bae_model, lvae_model, device):
    ckpt    = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    cfg     = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = device
    sampler = build_sampler(cfg)

    saved = ckpt["denoiser"]
    w_dim = cfg.lae_latent_dim
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
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device).eval()
    return ddm_w


def encode_train(bae_model, lvae_model, device):
    ds = NewWingsDataset(SLICES, SCALARS, seed=42)
    items = [x for x in ds["train"] if x["final"] == 1]
    w_list, coords_list = [], []
    lvae_ps = getattr(lvae_model, 'scaler_params', None)
    bae_model.eval(); lvae_model.encoder.eval()
    with torch.no_grad():
        for item in items:
            x = torch.tensor(item["coords"], dtype=torch.float32).unsqueeze(0).to(device)
            z = bae_model.encode(x)
            p = torch.tensor(item["coef_pressure"], dtype=torch.float32).unsqueeze(0).to(device)
            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            f_t  = torch.tensor(flow, dtype=torch.float32).unsqueeze(0).to(device)
            fv   = (torch.tensor(lvae_ps.transform(f_t), dtype=torch.float32).to(device)
                    if lvae_ps else f_t)
            w = lvae_model.encoder(z, p, fv).squeeze(0).cpu()
            w_list.append(w)
            coords_list.append(torch.tensor(item["coords"], dtype=torch.float32))
    return torch.stack(w_list), torch.stack(coords_list)


def mach_regime(mach):
    if mach < 0.8:
        return "subsonic"
    elif mach < 1.0:
        return "transonic"
    else:
        return "supersonic"


def shape_mse(a, b):
    return ((a - b) ** 2).mean().item()


def main():
    torch.manual_seed(0)
    cfg = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = DEVICE

    print("Loading models …")
    bae   = load_bae_3d(BAE_CKP, DEVICE)
    lvae  = load_lvae_3d(cfg, bae)
    ddm_w = load_ddm_w(bae, lvae, DEVICE)

    print("Encoding training set …")
    train_w, train_coords = encode_train(bae, lvae, DEVICE)   # [N_tr, 64], [N_tr, S, 2, 192]
    N_tr = train_w.shape[0]

    # Training self-NN distances computed after loading ckpt (below)

    print("Preparing test set …")
    ds = NewWingsDataset(SLICES, SCALARS, seed=42)
    all_test = list(ds["test"])
    initial_by_case = {x["case_num"]: x for x in all_test if x["initial"] == 1}
    test_final      = [x for x in all_test if x["final"] == 1]

    ckpt_raw = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    pms = ckpt_raw.get("params_mean_std")
    ams = ckpt_raw.get("aoas_mean_std")
    sc_params = scaler(pms) if pms else None
    sc_aoas   = scaler(ams) if ams else None

    gt_coords, gt_aoas, gt_pres, gt_w, gt_z, w_inits, params_norm, _ = precompute_test_3d(
        test_final, initial_by_case, bae, lvae,
        ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, DEVICE,
    )
    N_te = gt_coords.shape[0]
    machs = [float(x["mach"]) for x in test_final]

    print("Generating (averaging over passes) …")
    all_gen_coords = []
    all_gen_w      = []
    for _ in range(N_PASSES):
        with torch.no_grad():
            c, _, _, _, w, _ = ddm_w.generate(
                w_init=w_inits.to(DEVICE),
                params=params_norm.to(DEVICE),
                device=DEVICE,
            )
        all_gen_coords.append(c.cpu())
        # un-normalise w
        w_raw = w * ddm_w.w_std.squeeze(0) + ddm_w.w_mean.squeeze(0) if ddm_w.w_mean is not None else w
        all_gen_w.append(w_raw.cpu())

    gen_coords_mean = torch.stack(all_gen_coords).mean(0)   # [N_te, S, 2, 192]
    gen_w_mean      = torch.stack(all_gen_w).mean(0)        # [N_te, 64]

    print("Computing scatter metrics …")
    # Use only ACTIVE dimensions for distance (dead dims are near-constant
    # and inflate distances without carrying information).
    ckpt_loaded = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    w_std_vec   = ckpt_loaded.get("w_std", torch.ones(1, gen_w_mean.shape[1]))
    active_mask = (w_std_vec.squeeze() > 0.01)   # [64] bool, 20 True

    gen_w_act   = gen_w_mean[:, active_mask]      # [N_te, 20]
    train_w_act = train_w[:, active_mask]         # [N_tr, 20]

    # Training self-NN distances (active dims only, leave-one-out)
    d_tr = torch.cdist(train_w_act, train_w_act)  # [N_tr, N_tr]
    d_tr.fill_diagonal_(float('inf'))
    train_nn_dist = d_tr.min(dim=1).values.numpy()

    # Distance to nearest training neighbour (novelty)
    d_gen_tr = torch.cdist(gen_w_act, train_w_act)   # [N_te, N_tr]
    gen_nn_dist = d_gen_tr.min(dim=1).values.numpy()  # [N_te]

    # Shape MSE vs GT (fidelity)
    gen_mse = np.array([shape_mse(gen_coords_mean[i], gt_coords[i]) for i in range(N_te)])

    # Training set fidelity reference (encode + decode round-trip MSE is ~0,
    # use BAE round-trip vs coords as the floor)
    # For the training cloud we show the self-NN distance on x and a placeholder y=0
    # (training set by definition has 0 error against itself)

    print("Plotting …")
    regime_colors = {"subsonic": "#4477AA", "transonic": "#EE7733", "supersonic": "#CC3311"}
    regime_labels = {"subsonic": "Subsonic (M<0.8)", "transonic": "Transonic (0.8<=M<1.0)", "supersonic": "Supersonic (M>=1.0)"}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Training cloud: x = self-NN dist, y = 0 (reference, perfect fidelity)
    ax.scatter(train_nn_dist, np.zeros(N_tr),
               c="#bbbbbb", s=12, alpha=0.4, zorder=1, label="Training set (manifold)")

    # Generated points
    for regime, color in regime_colors.items():
        idx = [i for i, m in enumerate(machs) if mach_regime(m) == regime]
        if not idx:
            continue
        ax.scatter(gen_nn_dist[idx], gen_mse[idx],
                   c=color, s=40, alpha=0.85, zorder=3,
                   label=regime_labels[regime])

    # Vertical line: median training self-NN distance
    med_train = float(np.median(train_nn_dist))
    ax.axvline(med_train, color="#aaaaaa", lw=1.2, ls="--",
               label=f"Median train NN dist ({med_train:.2f})")

    ax.set_xlabel("Distance to nearest training-set neighbour (w-space)", fontsize=11)
    ax.set_ylabel("Shape MSE vs. ground-truth optimised wing", fontsize=11)
    ax.set_title("DDM-W: Fidelity vs. Novelty", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.00003)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")

    # Print summary
    print(f"\nGenerated: {N_te} wings")
    print(f"  Median NN dist (generated): {np.median(gen_nn_dist):.3f}")
    print(f"  Median NN dist (training):  {np.median(train_nn_dist):.3f}")
    print(f"  % generated OUTSIDE training cloud (NN dist > median training): "
          f"{100 * (gen_nn_dist > med_train).mean():.1f}%")
    print(f"  Median Shape MSE (generated): {np.median(gen_mse):.2e}")


if __name__ == "__main__":
    main()
