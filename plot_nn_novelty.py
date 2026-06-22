"""
Nearest-neighbour novelty plot for DDM-W.

For a handful of generated wings, find their 3 nearest neighbours in the
training set (Euclidean distance in LVAE w-space) and plot the generated
wing alongside those neighbours side-by-side.

Usage:
    python plot_nn_novelty.py
"""

import sys
import os
sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

import torch
import numpy as np
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans"})
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────────
BAE_CKP   = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP  = "results/lvae_3d/lvae_3d_v29_best.pth"
DDMW_CKP  = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"
OUT_PATH  = "thesis/figures/nn_novelty_ddmw.png"
DEVICE    = "cpu"

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

# ── imports ─────────────────────────────────────────────────────────────────
from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.lvae.train_lvae_3d import LVAE3D, LAEEncoderJoint3D
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.train_ddm_w_3d import load_bae_3d, load_lvae_3d, Config, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w_3d import precompute_test_3d
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

# matplotlib.pylab (imported transitively by unets.py) sets text.usetex=True on
# some systems. Override it here after all project imports are done.
plt.rcParams.update({"text.usetex": False})


def load_ddm_w(bae_model, lvae_model, device):
    ckpt    = torch.load(DDMW_CKP, map_location="cpu", weights_only=False)
    cfg     = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = device
    sampler = build_sampler(cfg)

    saved = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        w_dim = cfg.lae_latent_dim
        denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)

    pms = ckpt.get("params_mean_std")
    ams = ckpt.get("aoas_mean_std")

    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=cfg.lae_latent_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=pms, aoas_mean_std=ams,
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    ddm_w.denoiser.eval()
    print("DDM_W3D loaded.")
    return ddm_w


def encode_train_w(bae_model, lvae_model, device):
    """Encode all 767 training wings into w-space. Returns (w_all, coords_all)."""
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    train_items = [item for item in new_dataset["train"] if item["final"] == 1]
    print(f"Training wings: {len(train_items)}")

    lvae_ps = getattr(lvae_model, 'scaler_params', None)

    w_list, coords_list = [], []
    bae_model.eval()
    lvae_model.encoder.eval()

    with torch.no_grad():
        for item in train_items:
            x_wing = torch.tensor(item["coords"], dtype=torch.float32).unsqueeze(0).to(device)
            z_bae  = bae_model.encode(x_wing)

            pressure = torch.tensor(item["coef_pressure"], dtype=torch.float32).to(device)
            flow     = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_t   = torch.tensor(flow, dtype=torch.float32).unsqueeze(0).to(device)
            if lvae_ps is not None:
                flow_lvae = torch.tensor(
                    lvae_ps.transform(torch.tensor(flow, dtype=torch.float32).unsqueeze(0)),
                    dtype=torch.float32
                ).to(device)
            else:
                flow_lvae = flow_t

            if isinstance(lvae_model.encoder, LAEEncoderJoint3D):
                w = lvae_model.encoder(z_bae, pressure.unsqueeze(0), flow_lvae)
            else:
                w = lvae_model.encoder(z_bae, pressure.unsqueeze(0), flow_lvae)
            w_list.append(w.squeeze(0).cpu())

            coords_np = item["coords"]  # [S, 2, 192] or [S, 192, 2]
            coords_list.append(torch.tensor(coords_np, dtype=torch.float32))

    w_all      = torch.stack(w_list)      # [N_train, w_dim]
    coords_all = torch.stack(coords_list) # [N_train, ...]
    print(f"Training w encoded: {w_all.shape}")
    return w_all, coords_all


def generate_some(ddm_w, w_inits, params_norm, device, n_samples=5):
    """Generate n_samples wings using the first n_samples test conditions."""
    w_inits  = w_inits[:n_samples].to(device)
    params_n = params_norm[:n_samples].to(device)
    with torch.no_grad():
        coords, aoas, pressures, _, w_gen, _ = ddm_w.generate(
            w_init=w_inits, params=params_n, device=device
        )
    return coords.cpu(), w_gen.cpu()


def plot_slice(ax, coords, color, label=None, alpha=1.0):
    """Plot one spanwise slice (mid-span, index 7)."""
    # coords: [S, 2, 192] — upper/lower or [S, 192, 2]
    if coords.ndim == 3 and coords.shape[1] == 2:
        # [S, 2, 192]: slice s, xy 0=x 1=y, point k
        s_idx = coords.shape[0] // 2
        x = coords[s_idx, 0].numpy()
        y = coords[s_idx, 1].numpy()
    else:
        # [S, 192, 2]
        s_idx = coords.shape[0] // 2
        x = coords[s_idx, :, 0].numpy()
        y = coords[s_idx, :, 1].numpy()

    ax.plot(x, y, color=color, lw=1.2, alpha=alpha, label=label)
    ax.set_aspect("equal")
    ax.axis("off")


def make_plot(gen_coords_all, nn_coords_all, out_path):
    """
    gen_coords_all: list of [S, 2, 192] tensors (generated)
    nn_coords_all:  list of 3 × [S, 2, 192] tensors (nearest neighbours)
    """
    n = len(gen_coords_all)
    k = 3  # neighbours

    fig = plt.figure(figsize=(4 * (k + 1), 2.2 * n))
    gs  = gridspec.GridSpec(n, k + 1, figure=fig,
                            hspace=0.3, wspace=0.15)

    palette_nn = ["#7fbfff", "#3399ff", "#0066cc"]
    col_gen    = "#e05c00"

    for row, (gen_c, nn_cs) in enumerate(zip(gen_coords_all, nn_coords_all)):
        # Column 0: generated
        ax = fig.add_subplot(gs[row, 0])
        plot_slice(ax, gen_c, col_gen)
        if row == 0:
            ax.set_title("Generated", fontsize=9, color=col_gen, fontweight="bold")
        # ylabel doesn't render with axis("off") — use text annotation instead
        ax.text(-0.08, 0.5, f"#{row + 1}", transform=ax.transAxes,
                ha="right", va="center", fontsize=8, color="#444444")

        # Columns 1–3: nearest neighbours
        for col, (nn_c, clr) in enumerate(zip(nn_cs, palette_nn), start=1):
            ax = fig.add_subplot(gs[row, col])
            plot_slice(ax, nn_c, clr)
            if row == 0:
                ax.set_title(f"NN #{col}", fontsize=9, color=clr, fontweight="bold")

    fig.suptitle(
        "DDM-W: generated wings and their 3 nearest training-set neighbours\n"
        "(mid-span slice, Euclidean distance in w-space)",
        fontsize=10, y=1.01,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    torch.manual_seed(42)

    cfg = Config()
    cfg.bae_checkpoint  = BAE_CKP
    cfg.lvae_checkpoint = LVAE_CKP
    cfg.device          = DEVICE

    print("Loading BAE …")
    bae_model = load_bae_3d(BAE_CKP, DEVICE)

    print("Loading LVAE …")
    lvae_model = load_lvae_3d(cfg, bae_model)

    print("Loading DDM-W …")
    ddm_w = load_ddm_w(bae_model, lvae_model, DEVICE)

    # Encode training set
    print("Encoding training wings …")
    train_w, train_coords = encode_train_w(bae_model, lvae_model, DEVICE)

    # Prepare test set
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=42)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test wings: {len(test_dataset)}")

    pms = torch.load(DDMW_CKP, map_location="cpu", weights_only=False).get("params_mean_std")
    ams = torch.load(DDMW_CKP, map_location="cpu", weights_only=False).get("aoas_mean_std")
    scaler_params = scaler(pms) if pms is not None else None
    scaler_aoas   = scaler(ams) if ams is not None else None

    gt_coords, gt_aoas, gt_pressures, gt_w, gt_z, w_inits, params_norm, case_nums = \
        precompute_test_3d(
            test_dataset, initial_by_case, bae_model, lvae_model,
            ddm_w.w_mean, ddm_w.w_std, scaler_params, scaler_aoas, DEVICE,
        )

    # Generate 5 wings (pick spread-out test indices: 0, 20, 40, 60, 80)
    pick = [0, 20, 40, 60, 80]
    w_inits_pick  = w_inits[pick]
    params_pick   = params_norm[pick]

    print("Generating …")
    gen_coords, gen_w = generate_some(
        ddm_w, w_inits_pick, params_pick, DEVICE, n_samples=len(pick)
    )
    print(f"Generated coords: {gen_coords.shape}")

    # Un-normalise w for distance computation
    if ddm_w.w_mean is not None:
        gen_w_raw = gen_w * ddm_w.w_std.squeeze(0) + ddm_w.w_mean.squeeze(0)
    else:
        gen_w_raw = gen_w

    # Nearest neighbours in raw w-space
    dists = torch.cdist(gen_w_raw, train_w)  # [n_gen, N_train]
    nn_indices = dists.argsort(dim=1)[:, :3]  # [n_gen, 3]

    gen_coords_all = [gen_coords[i] for i in range(len(pick))]
    nn_coords_all  = [
        [train_coords[nn_indices[i, k]] for k in range(3)]
        for i in range(len(pick))
    ]

    print("Plotting …")
    make_plot(gen_coords_all, nn_coords_all, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
