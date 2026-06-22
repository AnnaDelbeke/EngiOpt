"""
Manifold Distance scatter plot for DDM-W (thesis-grade version).

X-axis: NN distance to training set, normalized by median training self-NN distance.
  x=1  → "average gap between two training wings" (on-manifold)
  x>1  → exploring novel territory
Y-axis: Shape MSE vs ground-truth optimised wing (fidelity).
Color:  Shape MSE mapped to a green→red colormap (green = low error = physically valid).

Training cloud shown as gray reference at y=0 (their own fidelity is 0 by definition).
"""

import sys, os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

# matplotlib.pylab (transitively imported by unets.py) sets text.usetex=True
plt.rcParams.update({"text.usetex": False})

# ── paths ───────────────────────────────────────────────────────────────────
BAE_CKP  = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"
OUT_PATH = "thesis/figures/manifold_distance_scatter.png"
SLICES   = "Wing_TL/data/processed/new_dataset_slices.pkl"
SCALARS  = "Wing_TL/data/processed/new_dataset_scalars.pkl"
DEVICE   = "cpu"
N_PASSES = 3


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
    ds    = NewWingsDataset(SLICES, SCALARS, seed=42)
    items = [x for x in ds["train"] if x["final"] == 1]
    w_list = []
    lvae_ps = getattr(lvae_model, 'scaler_params', None)
    bae_model.eval()
    lvae_model.encoder.eval()
    with torch.no_grad():
        for item in items:
            x  = torch.tensor(item["coords"],        dtype=torch.float32).unsqueeze(0).to(device)
            p  = torch.tensor(item["coef_pressure"], dtype=torch.float32).unsqueeze(0).to(device)
            z  = bae_model.encode(x)
            fl = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            ft = torch.tensor(fl, dtype=torch.float32).unsqueeze(0).to(device)
            fv = torch.tensor(lvae_ps.transform(ft), dtype=torch.float32).to(device) if lvae_ps else ft
            w  = lvae_model.encoder(z, p, fv).squeeze(0).cpu()
            w_list.append(w)
    return torch.stack(w_list)   # [N_tr, 64]


def mach_regime(m):
    return "subsonic" if m < 0.8 else ("transonic" if m < 1.0 else "supersonic")


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

    # Active-dimension mask (dead dims are zeroed by the LVAE mask and
    # contribute near-zero variance — exclude them from distances)
    w_std_vec   = ddm_w.w_std.squeeze()              # [64]
    active_mask = (w_std_vec > 0.01)                 # True for 20 active dims
    print(f"Active dims: {active_mask.sum().item()}")

    print("Encoding training set …")
    train_w_raw = encode_train(bae, lvae, DEVICE)    # [N_tr, 64], raw LVAE output
    train_w_act = train_w_raw[:, active_mask]        # [N_tr, 20]
    N_tr = train_w_act.shape[0]

    # Training self-NN distances (leave-one-out, active dims only)
    print("Computing training self-NN distances …")
    d_tr = torch.cdist(train_w_act, train_w_act)
    d_tr.fill_diagonal_(float("inf"))
    train_nn_dist = d_tr.min(dim=1).values.numpy()   # [N_tr]
    med_train     = float(np.median(train_nn_dist))
    print(f"Median training self-NN dist: {med_train:.3f}")

    print("Preparing test set …")
    ds   = NewWingsDataset(SLICES, SCALARS, seed=42)
    all_test        = list(ds["test"])
    initial_by_case = {x["case_num"]: x for x in all_test if x["initial"] == 1}
    test_final      = [x for x in all_test if x["final"] == 1]

    ckpt_raw  = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    sc_params = scaler(ckpt_raw.get("params_mean_std")) if ckpt_raw.get("params_mean_std") else None
    sc_aoas   = scaler(ckpt_raw.get("aoas_mean_std"))   if ckpt_raw.get("aoas_mean_std")   else None

    gt_coords, _, _, _, _, w_inits, params_norm, _ = precompute_test_3d(
        test_final, initial_by_case, bae, lvae,
        ddm_w.w_mean, ddm_w.w_std, sc_params, sc_aoas, DEVICE,
    )
    N_te  = gt_coords.shape[0]
    machs = [float(x["mach"]) for x in test_final]

    print(f"Generating {N_PASSES} passes over {N_te} test wings …")
    all_coords, all_w = [], []
    for i in range(N_PASSES):
        with torch.no_grad():
            # generate() returns w_raw (already un-normalized)
            c, _, _, _, w_raw, _ = ddm_w.generate(
                w_init=w_inits.to(DEVICE),
                params=params_norm.to(DEVICE),
                device=DEVICE,
            )
        all_coords.append(c.cpu())
        all_w.append(w_raw.cpu())
        print(f"  pass {i+1}/{N_PASSES} done")

    gen_coords = torch.stack(all_coords).mean(0)   # [N_te, S, 2, 192]
    gen_w_raw  = torch.stack(all_w).mean(0)        # [N_te, 64]
    gen_w_act  = gen_w_raw[:, active_mask]         # [N_te, 20]

    print("w_raw norm check — gen:", gen_w_act.norm(dim=1)[:5].tolist(),
          "train:", train_w_act.norm(dim=1)[:5].tolist())

    print("Computing scatter metrics …")
    # NN distance to training manifold, normalized by median training gap
    d_gen_tr    = torch.cdist(gen_w_act, train_w_act)          # [N_te, N_tr]
    gen_nn_raw  = d_gen_tr.min(dim=1).values.numpy()           # [N_te]
    gen_nn_norm = gen_nn_raw / med_train                        # normalized

    # Training self-NN distance, also normalized
    train_nn_norm = train_nn_dist / med_train                   # [N_tr], median ≈ 1.0

    # Shape MSE vs GT (fidelity)
    gen_mse = np.array([((gen_coords[i] - gt_coords[i])**2).mean().item()
                        for i in range(N_te)])

    print(f"Median normalized NN dist — gen: {np.median(gen_nn_norm):.2f}  "
          f"train (self): {np.median(train_nn_norm):.2f}")
    print(f"% gen with norm-dist > 1.0: {100*(gen_nn_norm > 1.0).mean():.1f}%")
    print(f"% gen with norm-dist > 1.5: {100*(gen_nn_norm > 1.5).mean():.1f}%")
    print(f"Median gen Shape MSE: {np.median(gen_mse):.2e}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("Plotting …")

    # Colormap: green (low MSE = good) → red (high MSE = bad)
    cmap   = plt.get_cmap("RdYlGn_r")
    vmin   = 0.0
    vmax   = float(np.percentile(gen_mse, 95))   # cap at 95th pct for contrast
    norm_c = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Gray reference cloud: training self-NN distances vs y=0 (perfect fidelity)
    ax.scatter(train_nn_norm, np.zeros(N_tr),
               c="#cccccc", s=10, alpha=0.35, zorder=1,
               label="Training set (y=0, reference)")

    # Generated points, colored by Shape MSE
    sc = ax.scatter(gen_nn_norm, gen_mse,
                    c=gen_mse, cmap=cmap, norm=norm_c,
                    s=55, alpha=0.90, edgecolors="#333333", linewidths=0.4,
                    zorder=3, label="Generated wings")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Shape MSE (green = low error)", fontsize=9)

    # Regime markers: overlay symbols on top of the colormap points
    markers = {"subsonic": "o", "transonic": "s", "supersonic": "^"}
    regime_colors = {"subsonic": "#4477AA", "transonic": "#EE7733", "supersonic": "#CC3311"}
    for regime, mkr in markers.items():
        idx = [i for i, m in enumerate(machs) if mach_regime(m) == regime]
        if not idx:
            continue
        ax.scatter(gen_nn_norm[idx], gen_mse[idx],
                   s=70, marker=mkr, facecolors="none",
                   edgecolors=regime_colors[regime], linewidths=1.5,
                   zorder=4,
                   label={"subsonic":   "Subsonic (M<0.8)",
                          "transonic":  "Transonic (0.8-1.0)",
                          "supersonic": "Supersonic (M>=1.0)"}[regime])

    # Reference lines
    ax.axvline(1.0, color="#888888", lw=1.2, ls="--", label="x=1: median training gap")
    ax.axvline(1.5, color="#aaaaaa", lw=0.9, ls=":",  label="x=1.5: novel territory")

    # Annotation bands
    ax.axvspan(0, 1.0,  alpha=0.04, color="blue",  zorder=0)
    ax.axvspan(1.0, 1.5, alpha=0.04, color="gray", zorder=0)
    ax.axvspan(1.5, ax.get_xlim()[1] if ax.get_xlim()[1] > 1.5 else 3.0,
               alpha=0.06, color="orange", zorder=0)

    ax.set_xlabel("Normalized distance to nearest training neighbour  "
                  "(1 = median training gap)", fontsize=11)
    ax.set_ylabel("Shape MSE vs. ground-truth optimised wing", fontsize=11)
    ax.set_title("DDM-W: Novelty vs. Fidelity\n"
                 "Color = Shape MSE (green: low error, red: high error)  "
                 "|  Marker = Mach regime", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.000005)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
