"""
Initialization sensitivity analysis for 3D generative models.

For a set of anchor flow conditions, generate predictions conditioned on
K different initial wings and measure how much the outputs diverge.

Usage
-----
    # DDM_W3D
    python -m engiopt.analysis.sensitivity_analysis_3d \
        --model ddm_w \
        --checkpoint      results/ddm_w_3d/ddm_w_3d_v1_best.pth \
        --bae_checkpoint  results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v8_best.pth

    # DDM-3D
    python -m engiopt.analysis.sensitivity_analysis_3d \
        --model ddm_3d \
        --checkpoint     results/ddm_3d/ddm_3d_v1_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt
"""

import argparse
import json
import os
from datetime import datetime, timezone
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex":          False,
    "text.latex.preamble":  "",
    "font.family":          "DejaVu Sans",
    "mathtext.default":     "regular",
    "pdf.use14corefonts":   True,
})
import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler
from engiopt.lvae.train_lvae_3d import LAEEncoderJoint3D

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_ANCHORS   = 10   # flow conditions to analyse (spread across regimes)
N_INITS     = 10   # different initial wings per anchor
N_PASSES    = 3    # diffusion passes per (anchor, init) pair


# ---------------------------------------------------------------------------
# Coordinate normalisation (same as eval scripts)
# ---------------------------------------------------------------------------

def normalise_coords(coords, te_shifts):
    """coords [S,192,2], te_shifts [S] → normalised [S,2,192] + le_x, chord"""
    c = coords.clone()
    c[:, :, 1] -= te_shifts.unsqueeze(1)
    te_x  = c[:, 0, 0]
    c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
    le_x  = c[:, :, 0].min(dim=1).values
    chord = 1.0 - le_x
    c[:, :, 0] = (c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)
    return c.permute(0, 2, 1)   # [S, 2, 192]


def encode_wing(item, bae_model, device):
    """Return z_bae [1, latent_dim] for a dataset item."""
    coords    = torch.tensor(item["coords"],    dtype=torch.float32)
    te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
    x = normalise_coords(coords, te_shifts).unsqueeze(0).to(device)
    return bae_model.encode(x)   # [1, latent_dim]


# ---------------------------------------------------------------------------
# Init-encoding helpers (model-specific)
# ---------------------------------------------------------------------------

def encode_init_ddm_w(init_item, bae_model, lvae_model, flow_lvae, w_mean, w_std, device):
    """Encode an initial wing into normalised w space for DDM_W3D."""
    z_init = encode_wing(init_item, bae_model, device)
    is_joint = isinstance(lvae_model.encoder, LAEEncoderJoint3D)
    if is_joint:
        pressure = torch.tensor(init_item["coef_pressure"], dtype=torch.float32).unsqueeze(0).to(device)
        w_init = lvae_model.encoder(z_init, pressure, flow_lvae).squeeze(0).cpu()
    else:
        w_init = lvae_model.encoder(z_init, flow_lvae).squeeze(0).cpu()
    return (w_init - w_mean.squeeze(0)) / w_std.squeeze(0)   # [w_dim]


def encode_init_ddm_3d(init_item, bae_model, z_mean, z_std, device):
    """Encode an initial wing into normalised z space for DDM-3D."""
    z_init = encode_wing(init_item, bae_model, device).squeeze(0).cpu()
    if z_mean is not None:
        return (z_init - z_mean) / z_std
    return z_init


# ---------------------------------------------------------------------------
# Generation wrappers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ddm_w(ddm_w, w_init_norm, params_norm, device, n_passes):
    """Run n_passes and average. Returns coords [S,2,192], aoa scalar."""
    w_in = w_init_norm.squeeze().unsqueeze(0).to(device)   # [1, w_dim]
    p_in = params_norm.squeeze().unsqueeze(0).to(device)   # [1, c_dim]
    coords_list, aoas_list = [], []
    for _ in range(n_passes):
        coords, aoas, *_ = ddm_w.generate(w_init=w_in, params=p_in, device=device)
        coords_list.append(coords.squeeze(0))
        aoas_list.append(aoas.squeeze(0))
    coords_mean = torch.stack(coords_list).mean(0)   # [S, 2, 192]
    aoa_mean    = torch.stack(aoas_list).mean(0)
    return coords_mean, float(aoa_mean)


@torch.no_grad()
def generate_ddm_3d(model, z_init_norm, params_norm, device, n_passes):
    """Run n_passes and average. Returns coords [S,2,192], aoa scalar."""
    from engiopt.ddm.evaluate_ddm_3d import generate
    # ensure [1, dim] — squeeze first to drop any extra dims, then unsqueeze
    z_in = z_init_norm.squeeze().unsqueeze(0)   # [1, z_dim]
    p_in = params_norm.squeeze().unsqueeze(0)   # [1, c_dim]
    coords_list, aoas_list = [], []
    for _ in range(n_passes):
        coords, aoa = generate(model, z_in, p_in, device)
        coords_list.append(coords.squeeze(0))
        aoas_list.append(float(aoa.squeeze()))
    return torch.stack(coords_list).mean(0), float(np.mean(aoas_list))


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

def pairwise_shape_mse(coords_list):
    """Mean pairwise MSE across all pairs. coords_list: list of [S,2,192]."""
    n = len(coords_list)
    if n < 2:
        return 0.0
    vals = []
    for a, b in combinations(range(n), 2):
        vals.append(((coords_list[a] - coords_list[b]) ** 2).mean().item())
    return float(np.mean(vals))


def aoa_std(aoas):
    return float(np.std(aoas))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wing_overlays(results, save_path, model_name):
    """
    For one representative anchor per flow regime, plot all generated wings
    overlaid on each other — one slice per column, one regime per row.
    Shows visually how much outputs diverge across initializations.
    """
    regime_order  = ["subsonic", "transonic", "supersonic"]
    regime_colors = {"subsonic": "steelblue", "transonic": "darkorange", "supersonic": "firebrick"}
    regime_labels = {"subsonic": "Subsonic", "transonic": "Transonic", "supersonic": "Supersonic"}

    def _get_coords(r):
        if "_coords_list" in r:
            return [c.numpy() if hasattr(c, 'numpy') else np.array(c) for c in r["_coords_list"]]
        elif "coords_list" in r:
            return [np.array(c) for c in r["coords_list"]]
        return None

    def _get_gt(r):
        if "_gt_coords" in r:
            gt = r["_gt_coords"]
            return gt.numpy() if hasattr(gt, 'numpy') else np.array(gt)
        elif "gt_coords" in r:
            return np.array(r["gt_coords"])
        return None

    # Pick one representative anchor per regime (highest pairwise MSE = most interesting)
    rep = {}
    for regime in regime_order:
        candidates = [r for r in results if r["regime"] == regime and _get_coords(r) is not None]
        if candidates:
            rep[regime] = max(candidates, key=lambda r: r["pairwise_shape_mse"])

    slice_indices = [0, 4, 8, 14]   # root, mid, near-tip, tip
    n_regimes = len(rep)
    n_slices  = len(slice_indices)

    fig, axes = plt.subplots(n_regimes, n_slices,
                             figsize=(n_slices * 3.5, n_regimes * 3.0))
    if n_regimes == 1:
        axes = axes[np.newaxis, :]

    for row, regime in enumerate(regime_order):
        if regime not in rep:
            continue
        r = rep[regime]
        coords_list = _get_coords(r)
        color = regime_colors[regime]

        gt = _get_gt(r)
        for col, s_idx in enumerate(slice_indices):
            ax = axes[row, col]
            for coords in coords_list:
                sl = coords[s_idx]
                ax.plot(sl[0], sl[1], color=color, alpha=0.35, linewidth=0.8)
            if gt is not None:
                sl = gt[s_idx]
                ax.plot(sl[0], sl[1], color='black', linewidth=1.4,
                        linestyle='--', zorder=5, label='GT' if col == 0 else None)
            ax.set_aspect('equal')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"Slice {s_idx}", fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"{regime_labels[regime]}\nM={r['mach']:.2f}\ncase {r['case_num']}",
                    fontsize=8, rotation=0, labelpad=60, va='center'
                )
                if gt is not None:
                    ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f"{model_name} - Generated wings per initialization (overlaid)", fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.05, hspace=0.1, wspace=0.05)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_gt_mse(results, save_path, model_name):
    """
    Box plot + strip plot of GT-MSE, one box per flow regime, pooling all
    initializations across all anchors of that regime.
    """
    regime_order  = ["subsonic", "transonic", "supersonic"]
    regime_colors = {"subsonic": "steelblue", "transonic": "darkorange", "supersonic": "firebrick"}
    regime_labels = {"subsonic": "Subsonic\n(M < 0.8)",
                     "transonic": "Transonic\n(0.8 - 1.0)",
                     "supersonic": "Supersonic\n(M >= 1.0)"}

    # Pool all gt_mse values across all anchors per regime
    data, labels, colors = [], [], []
    for regime in regime_order:
        vals = []
        for r in results:
            if r["regime"] == regime and "gt_mse_list" in r:
                vals.extend(r["gt_mse_list"])
        if vals:
            data.append(vals)
            labels.append(regime_labels[regime])
            colors.append(regime_colors[regime])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(len(data) * 3.0 + 1.5, 5.0))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.45, notch=False,
                    medianprops=dict(color='black', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='', linestyle='none'))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.7, s=18, zorder=3, edgecolors='none')

    ax.set_ylabel("MSE vs ground truth wing", fontsize=10)
    ax.set_title(f"{model_name} - GT-MSE per initialization by flow regime", fontsize=11)
    ax.tick_params(axis='x', labelsize=9)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.90)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sensitivity(results, save_path):
    """
    Bar chart of mean pairwise shape MSE per anchor, coloured by flow regime.
    """
    labels  = [f"case {r['case_num']}\nM={r['mach']:.2f}" for r in results]
    mses    = [r['pairwise_shape_mse'] for r in results]
    aoa_stds = [r['aoa_std'] for r in results]
    colors  = []
    for r in results:
        m = r['mach']
        if m < 0.8:
            colors.append('steelblue')
        elif m < 1.0:
            colors.append('darkorange')
        else:
            colors.append('firebrick')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(labels, mses, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Mean pairwise shape MSE", fontsize=10)
    ax.set_title("Initialization sensitivity - shape", fontsize=11)
    ax.tick_params(axis='x', labelsize=7)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue',  label='Subsonic (M<0.8)'),
        Patch(facecolor='darkorange', label='Transonic (0.8-1.0)'),
        Patch(facecolor='firebrick',  label='Supersonic (M>=1.0)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax = axes[1]
    ax.bar(labels, aoa_stds, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("AoA std across initializations (deg)", fontsize=10)
    ax.set_title("Initialization sensitivity - AoA", fontsize=11)
    ax.tick_params(axis='x', labelsize=7)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.92, wspace=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sensitivity_by_regime(results, save_path):
    """Box plot of pairwise shape MSE grouped by flow regime."""
    regimes = {
        'Subsonic (M<0.8)':    [r['pairwise_shape_mse'] for r in results if r['mach'] < 0.8],
        'Transonic (0.8-1.0)': [r['pairwise_shape_mse'] for r in results if 0.8 <= r['mach'] < 1.0],
        'Supersonic (M>=1.0)': [r['pairwise_shape_mse'] for r in results if r['mach'] >= 1.0],
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [v for v in regimes.values() if v]
    labels = [k for k, v in regimes.items() if v]
    colors = ['steelblue', 'darkorange', 'firebrick'][:len(data)]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean pairwise shape MSE", fontsize=10)
    ax.set_title("Sensitivity by flow regime", fontsize=11)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.92)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           type=str, required=True, choices=["ddm_w", "ddm_3d"])
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, default=None,
                   help="Required for --model ddm_w")
    p.add_argument("--n_anchors",  type=int, default=N_ANCHORS)
    p.add_argument("--n_inits",    type=int, default=N_INITS)
    p.add_argument("--n_passes",   type=int, default=N_PASSES)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out_dir",    type=str, default="results/sensitivity_3d")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Load BAE ──────────────────────────────────────────────────────────────
    from engiopt.ddm.train_ddm_3d import load_bae_3d
    bae_model = load_bae_3d(args.bae_checkpoint, device, n_spans=15)
    bae_model.eval()

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.model == "ddm_w":
        assert args.lvae_checkpoint, "--lvae_checkpoint required for ddm_w"
        from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
        from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
        from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, load_lvae_3d, build_sampler

        cfg = Config()
        cfg.bae_checkpoint  = args.bae_checkpoint
        cfg.lvae_checkpoint = args.lvae_checkpoint
        cfg.device = device

        lvae_model = load_lvae_3d(cfg, bae_model)
        sampler    = build_sampler(cfg)
        w_dim      = cfg.lae_latent_dim
        pms = ckpt.get("params_mean_std")
        ams = ckpt.get("aoas_mean_std")

        saved = ckpt["denoiser"]
        if isinstance(saved, MLPDenoiser):
            denoiser = saved
        else:
            denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
            denoiser.load_state_dict(saved)

        ddm_model = DDM_W3D(
            denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
            sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
            w_pressure=ckpt.get("w_pressure", 1.0),
            w_aoa=ckpt.get("w_aoa", 1.0),
            lvae_params_dim=cfg.c_dim,
            params_mean_std=pms, aoas_mean_std=ams,
            name="sensitivity",
        )
        ddm_model.w_mean     = ckpt.get("w_mean")
        ddm_model.w_std      = ckpt.get("w_std")
        ddm_model.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
        ddm_model.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
        ddm_model.denoiser.to(device)
        scaler_params = scaler(pms) if pms is not None else None
        w_mean = ddm_model.w_mean
        w_std  = ddm_model.w_std
        lvae_ps = getattr(lvae_model, 'scaler_params', None)
        print("DDM_W3D loaded.")

    else:  # ddm_3d
        from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
        from engiopt.ddm.train_ddm_3d import DDM3D
        from engiopt.ddm import samplers

        pms = ckpt.get("params_mean_std")
        ams = ckpt.get("aoas_mean_std")
        z_dim = ckpt.get("z_dim", 64)

        sampler  = samplers.BaselineSampler_AoA_3D(
            1000, start_x=1e-4, end_x=0.02, start_alpha=1e-4, end_alpha=0.02,
        )
        denoiser = MLPDenoiser(w_dim=z_dim, c_dim=4)
        denoiser.load_state_dict(ckpt["denoiser"])
        denoiser.to(device).eval()

        ddm_model = DDM3D(
            denoiser=denoiser, bae_model=bae_model, sampler=sampler,
            z_dim=z_dim, c_dim=4, params_mean_std=pms, aoas_mean_std=ams,
            name="sensitivity",
        )
        ddm_model.z_mean = ckpt.get("z_mean")
        ddm_model.z_std  = ckpt.get("z_std")
        scaler_params = scaler(pms) if pms is not None else None
        lvae_model = None
        print("DDM-3D loaded.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test = list(dataset["test"])
    test_finals   = [item for item in all_test if item["final"]   == 1]
    test_initials = [item for item in all_test if item["initial"] == 1]

    # Select anchor cases: spread across flow regimes
    machs = np.array([item["mach"] for item in test_finals])
    subsonic    = [i for i, m in enumerate(machs) if m < 0.8]
    transonic   = [i for i, m in enumerate(machs) if 0.8 <= m < 1.0]
    supersonic  = [i for i, m in enumerate(machs) if m >= 1.0]

    rng = np.random.default_rng(args.seed)
    n_sub  = max(1, args.n_anchors * len(subsonic)   // len(test_finals))
    n_tra  = max(1, args.n_anchors * len(transonic)  // len(test_finals))
    n_sup  = max(1, args.n_anchors * len(supersonic) // len(test_finals))
    # adjust to exactly n_anchors
    while n_sub + n_tra + n_sup < args.n_anchors:
        n_sub += 1
    anchor_idxs = (
        rng.choice(subsonic,   min(n_sub, len(subsonic)),   replace=False).tolist() +
        rng.choice(transonic,  min(n_tra, len(transonic)),  replace=False).tolist() +
        rng.choice(supersonic, min(n_sup, len(supersonic)), replace=False).tolist()
    )
    anchor_cases = [test_finals[i] for i in anchor_idxs]
    print(f"Anchors: {len(anchor_cases)} cases  |  Inits per anchor: {args.n_inits}")

    # ── Run sensitivity analysis ───────────────────────────────────────────────
    results = []

    for anchor in anchor_cases:
        case_num = anchor["case_num"]
        mach     = anchor["mach"]
        print(f"\nAnchor case {case_num}  Mach={mach:.3f}  Re={anchor['reynolds']:.2e}")

        # Flow params for this anchor
        flow = [anchor["mach"], anchor["reynolds"], anchor["cl_target"], anchor["area_case_ratio"]]
        flow_t = torch.tensor(flow, dtype=torch.float32)
        if scaler_params is not None:
            params_norm = scaler_params.transform(flow_t.unsqueeze(0)).squeeze()
        else:
            params_norm = flow_t

        # For DDM_W: also need LVAE-normalised flow params
        if args.model == "ddm_w":
            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            flow_lvae_np = lvae_ps.transform(flow_np) if lvae_ps is not None else flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)

        # Pick N_INITS different initial wings (exclude same case)
        pool = [item for item in test_initials if item["case_num"] != case_num]
        chosen_inits = rng.choice(len(pool), min(args.n_inits, len(pool)), replace=False)
        init_items = [pool[i] for i in chosen_inits]

        coords_list = []
        aoas_list   = []

        for init_item in init_items:
            if args.model == "ddm_w":
                w_init_norm = encode_init_ddm_w(
                    init_item, bae_model, lvae_model, flow_lvae,
                    w_mean, w_std, device
                )
                coords, aoa = generate_ddm_w(
                    ddm_model, w_init_norm, params_norm, device, args.n_passes
                )
            else:
                z_init_norm = encode_init_ddm_3d(
                    init_item, bae_model,
                    ddm_model.z_mean, ddm_model.z_std, device
                )
                coords, aoa = generate_ddm_3d(
                    ddm_model, z_init_norm, params_norm, device, args.n_passes
                )
            coords_list.append(coords)
            aoas_list.append(aoa)
            print(f"  init case {init_item['case_num']} done  aoa={aoa:.2f}")

        pw_mse = pairwise_shape_mse(coords_list)
        a_std  = aoa_std(aoas_list)

        gt_coords = normalise_coords(
            torch.tensor(anchor["coords"],    dtype=torch.float32),
            torch.tensor(anchor["te_shifts"], dtype=torch.float32),
        )  # [S, 2, 192]

        gt_mse_list = [
            float(((c - gt_coords) ** 2).mean().item()) for c in coords_list
        ]
        mean_gt_mse = float(np.mean(gt_mse_list))

        regime = "subsonic" if mach < 0.8 else ("transonic" if mach < 1.0 else "supersonic")
        print(f"  → pairwise MSE: {pw_mse:.4e}  gt_mse: {mean_gt_mse:.4e}  aoa_std: {a_std:.3f}°  [{regime}]")

        results.append({
            "case_num":           case_num,
            "mach":               mach,
            "reynolds":           anchor["reynolds"],
            "cl_target":          anchor["cl_target"],
            "regime":             regime,
            "pairwise_shape_mse": pw_mse,
            "mean_gt_mse":        mean_gt_mse,
            "gt_mse_list":        gt_mse_list,
            "aoa_std":            a_std,
            "n_inits":            len(init_items),
            "_coords_list":       coords_list,
            "_coords_list_json":  [c.tolist() for c in coords_list],
            "_gt_coords":         gt_coords,
            "_gt_coords_json":    gt_coords.tolist(),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Sensitivity Summary ===")
    for regime in ["subsonic", "transonic", "supersonic"]:
        sub = [r for r in results if r["regime"] == regime]
        if not sub:
            continue
        mean_pw  = np.mean([r["pairwise_shape_mse"] for r in sub])
        mean_gt  = np.mean([r["mean_gt_mse"]        for r in sub])
        mean_aoa = np.mean([r["aoa_std"]             for r in sub])
        print(f"  {regime:12s}: pairwise={mean_pw:.4e}  gt_mse={mean_gt:.4e}  aoa_std={mean_aoa:.3f}°  (n={len(sub)})")

    overall_mse = np.mean([r["pairwise_shape_mse"] for r in results])
    overall_gt  = np.mean([r["mean_gt_mse"]        for r in results])
    overall_aoa = np.mean([r["aoa_std"]             for r in results])
    print(f"  {'overall':12s}: pairwise={overall_mse:.4e}  gt_mse={overall_gt:.4e}  aoa_std={overall_aoa:.3f}°")

    # ── Save ─────────────────────────────────────────────────────────────────
    stem = f"sensitivity_{args.model}_{ts}"
    out  = {
        "model":      args.model,
        "checkpoint": args.checkpoint,
        "n_anchors":  len(results),
        "n_inits":    args.n_inits,
        "n_passes":   args.n_passes,
        "overall_pairwise_shape_mse": overall_mse,
        "overall_mean_gt_mse":        overall_gt,
        "overall_aoa_std":            overall_aoa,
        "per_anchor": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"coords_list": r["_coords_list_json"], "gt_coords": r["_gt_coords_json"]}
            for r in results
        ],
    }
    with open(os.path.join(args.out_dir, stem + ".json"), "w") as f:
        json.dump(out, f, indent=2)

    import traceback
    for fn, label in [
        (lambda: plot_sensitivity(results,           os.path.join(args.out_dir, stem + "_bars.png")),    "bars"),
        (lambda: plot_sensitivity_by_regime(results, os.path.join(args.out_dir, stem + "_regime.png")), "regime"),
        (lambda: plot_wing_overlays(results,         os.path.join(args.out_dir, stem + "_wings.png"),
                                    model_name=args.model.upper()),                                      "wings"),
        (lambda: plot_gt_mse(results,                os.path.join(args.out_dir, stem + "_gt_mse.png"),
                              model_name=args.model.upper()),                                            "gt_mse"),
    ]:
        try:
            fn()
            print(f"  [{label}] saved")
        except Exception:
            print(f"  [{label}] FAILED:")
            traceback.print_exc()

    print(f"Results saved to {args.out_dir}/{stem}.json")


if __name__ == "__main__":
    main()
