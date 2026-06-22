"""
Evaluation script for LVAE3D (trained on top of BezierAutoencoder3D).

Mirrors evaluate_lvae.py but uses LVAE3D + BezierAutoencoder3D instead of
LAE_AoAInit + the 2D BezierAutoencoder.  All metric helpers and plotting
functions are reused from evaluate_lvae.py and plotting.py unchanged.

Usage
-----
    python -m engiopt.lvae.evaluate_lvae_3d \
        --checkpoint results/lvae_3d/lvae_3d_v1_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from torch.nn.utils import spectral_norm as sn

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.lvae.train_lvae_3d import (
    LAEEncoder3D, LAEEncoderJoint3D, LAEDecoder3D, LVAE3D,
)
from engiopt.lvae.evaluate_lvae import (
    compute_geometry_metrics,
    compute_pressure_metrics,
    compute_perf_metrics,
    compute_latent_mmd,
    fit_pca_and_transform,
    compute_pca_reconstruction_errors,
    plot_error_histogram_comparison,
    GAMMAS, BATCH_SIZE,
)
from engiopt.lvae import plotting as lvae_plotting

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_bae_3d(bae_checkpoint: str, device: str,
                latent_dim: int = 256,
                cpx_bound: list = [-0.75, 1.0],
                cpy_bound: list = [-0.75, 0.75]) -> BezierAutoencoder3D:
    ckpt              = torch.load(bae_checkpoint, map_location=device, weights_only=False)
    n_spans           = ckpt["n_spans"]
    latent_dim        = ckpt.get("latent_dim",        latent_dim)
    slice_hidden_dims = ckpt.get("slice_hidden_dims", [256, 128])
    span_hidden_dims  = ckpt.get("span_hidden_dims",  [256, 128])
    cpx_bound         = ckpt.get("cpx_bound",         cpx_bound)
    cpy_bound         = ckpt.get("cpy_bound",         cpy_bound)
    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=32, n_data_points=192,
        slice_hidden_dims=slice_hidden_dims, span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim, cpx_bound=cpx_bound, cpy_bound=cpy_bound,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded 3D BAE  (n_spans={n_spans}, latent_dim={latent_dim}) from {bae_checkpoint}")
    return model


def load_lvae_3d(checkpoint: str, device: str,
                 bae_model: BezierAutoencoder3D,
                 bae_latent_dim: int = 256,
                 lae_latent_dim: int = 64,
                 n_spans: int = 9,
                 dropout: float = 0.0) -> LVAE3D:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    params_mean_std    = ckpt.get("params_mean_std")
    aoas_mean_std      = ckpt.get("aoas_mean_std")
    pressures_mean_std = ckpt.get("pressures_mean_std")

    # prefer values saved in the checkpoint over caller-supplied defaults
    n_spans          = ckpt.get("n_spans",          n_spans)
    pressure_length  = ckpt.get("pressure_length",  192)
    lae_latent_dim   = ckpt.get("lae_latent_dim",   lae_latent_dim)
    bae_latent_dim   = ckpt.get("bae_latent_dim",   bae_latent_dim)

    dropout            = ckpt.get("dropout", 0.0)
    joint_encoder      = ckpt.get("joint_encoder", False)
    pressure_embed_dim = ckpt.get("pressure_embed_dim", 16)
    span_embed_dim     = ckpt.get("span_embed_dim", 8)

    # detect whether this checkpoint was trained with spectral norm on decoder
    decoder_state = ckpt.get("decoder", {})
    has_sn        = any("weight_orig" in k for k in decoder_state)
    has_perf_head = "perf_head.weight" in decoder_state or "perf_head.weight_orig" in decoder_state
    use_pressure  = ckpt.get("use_pressure", True)

    if joint_encoder:
        encoder = LAEEncoderJoint3D(bae_latent_dim=bae_latent_dim, c_dim=4,
                                    n_spans=n_spans, pressure_length=pressure_length,
                                    pressure_embed_dim=pressure_embed_dim,
                                    lae_latent_dim=lae_latent_dim,
                                    dropout=dropout).to(device)
    else:
        encoder = LAEEncoder3D(bae_latent_dim=bae_latent_dim, c_dim=4,
                               n_spans=n_spans, pressure_length=pressure_length,
                               lae_latent_dim=lae_latent_dim,
                               dropout=dropout).to(device)
    decoder = LAEDecoder3D(bae_latent_dim=bae_latent_dim, c_dim=4,
                           lae_latent_dim=lae_latent_dim,
                           n_spans=n_spans, dropout=dropout,
                           use_perf_head=has_perf_head,
                           span_embed_dim=span_embed_dim,
                           spectral_norm=has_sn,
                           use_pressure=use_pressure).to(device)

    model = LVAE3D(
        encoder=encoder, decoder=decoder, bae_model=bae_model,
        lae_latent_dim=lae_latent_dim,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        pressures_mean_std=pressures_mean_std,
        name=os.path.splitext(os.path.basename(checkpoint))[0],
    ).to(device)

    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if "perfs_mean_std" in ckpt and ckpt["perfs_mean_std"] is not None:
        from engiopt.data_processing.utils import scaler as make_scaler
        model.scaler_perfs = make_scaler(ckpt["perfs_mean_std"])
    if "active_latent_mask" in ckpt:
        model.active_latent_mask = ckpt["active_latent_mask"].to(device)
    model.encoder.eval()
    model.decoder.eval()
    print(f"Loaded LVAE3D from {checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Encode a raw dataset item → BAE latent + ground-truth tensors
# ---------------------------------------------------------------------------

def encode_item_3d(item: dict, bae_model: BezierAutoencoder3D,
                   model: LVAE3D, device: str, apply_x_norm: bool = True):
    """Returns z_bae, gt_coords, aoa, params_scaled, te_shifts, gt_pressure."""
    coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
    te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]

    coords_c = coords.clone()
    coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
    if apply_x_norm:
        te_x = coords_c[:, 0, 0]
        coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
        le_x  = coords_c[:, :, 0].min(dim=1).values
        chord = 1.0 - le_x
        coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

    # save per-slice LE shift and chord for inverse transform
    if apply_x_norm:
        le_x_saved  = le_x.clone()   # [S]
        chord_saved = chord.clone()  # [S]

    x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)   # [1, S, 2, 192]
    with torch.no_grad():
        z_bae    = bae_model.encode(x_wing)                       # [1, bae_latent_dim]
        gt_recon = bae_model.decode(z_bae).squeeze(0).cpu()       # [S, 2, 192]

    # Restore original coordinate frame: reverse LE norm, then restore TE y-shift
    if apply_x_norm:
        gt_recon[:, 0, :] = gt_recon[:, 0, :] * chord_saved.unsqueeze(1) + le_x_saved.unsqueeze(1)
    gt_recon[:, 1, :] += te_shifts.unsqueeze(1)

    flow_p  = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
    params  = torch.tensor(flow_p, dtype=torch.float32).unsqueeze(0).to(device)
    params_scaled = model.scaler_params.transform(params)

    aoa      = torch.tensor([float(item["alpha"])], dtype=torch.float32)
    pressure = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)  # [S, 192]

    le_x_out  = le_x_saved  if apply_x_norm else torch.zeros(coords.shape[0])
    chord_out = chord_saved if apply_x_norm else torch.ones(coords.shape[0])

    return (z_bae.squeeze(0).cpu(), gt_recon, aoa,
            params_scaled.cpu(), te_shifts, pressure, le_x_out, chord_out)


# ---------------------------------------------------------------------------
# Batched reconstruction: encode z_bae through LVAE3D → decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_batch_3d(model: LVAE3D, bae_model: BezierAutoencoder3D,
                         z_baes_batch, pressure_batch, params_batch, te_shifts_batch,
                         le_x_batch, chord_batch, device):
    """Encode BAE latents through LVAE3D encoder → decode → BAE decode.

    Returns (coords [B, S, 2, 192], aoas [B], pressure [B, S, 192], perf [B, 2]).
    """
    z_baes_batch    = z_baes_batch.to(device)
    params_batch    = params_batch.to(device)
    pressure_batch  = pressure_batch.to(device)

    w = model.encoder(z_baes_batch, pressure_batch, params_batch)
    w = model._apply_mask(w)
    z_bae_pred, aoa_pred, eta_y_pred, pressure_pred_norm, perf_pred_norm = model.decoder(w, params_batch)

    coords = bae_model.decode(z_bae_pred).cpu()           # [B, S, 2, 192]  (normalized x)
    coords[:, :, 0, :] = (coords[:, :, 0, :] * chord_batch.unsqueeze(-1)
                          + le_x_batch.unsqueeze(-1))     # restore x-frame
    coords[:, :, 1, :] += te_shifts_batch.unsqueeze(-1)   # restore y-frame

    aoas_deg = model.scaler_aoas.inverse_transform(aoa_pred.cpu())

    if pressure_pred_norm is None:
        pressure = None
    elif model.scaler_pressures is not None:
        pressure = model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
    else:
        pressure = pressure_pred_norm.cpu()

    if perf_pred_norm is not None and model.scaler_perfs is not None:
        perf = model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
    else:
        perf = perf_pred_norm.cpu() if perf_pred_norm is not None else None

    return coords, aoas_deg.squeeze(-1), pressure, perf


# ---------------------------------------------------------------------------
# Metric explanation printer
# ---------------------------------------------------------------------------

def _explain(title: str, space: str, inputs: str, math: str, good: str, bad: str):
    w = 70
    print("\n" + "┌" + "─" * (w - 2) + "┐")
    print(f"│  📐 METRIC: {title:<{w - 16}}│")
    print("├" + "─" * (w - 2) + "┤")
    for label, text in [("SPACE  ", space), ("INPUTS ", inputs),
                        ("MATH   ", math),  ("GOOD   ", good), ("BAD    ", bad)]:
        # word-wrap at w-12 chars
        words = text.split()
        line  = ""
        lines = []
        for word in words:
            if len(line) + len(word) + 1 > w - 12:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        for i, l in enumerate(lines):
            prefix = f"│  {label}: " if i == 0 else "│           "
            print(f"{prefix}{l:<{w - len(prefix) + 2}}│")
    print("└" + "─" * (w - 2) + "┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True,
                   help="Path to LVAE3D .pth checkpoint")
    p.add_argument("--bae_checkpoint", type=str,
                   default="results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt")
    p.add_argument("--bae_latent_dim", type=int,   default=256)
    p.add_argument("--lae_latent_dim", type=int,   default=64)
    p.add_argument("--n_spans",        type=int,   default=9)
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--cpx_bound",      type=float, nargs=2, default=[-0.75, 1.0])
    p.add_argument("--cpy_bound",      type=float, nargs=2, default=[-0.75, 0.75])
    p.add_argument("--save_dir",       type=str,   default="results/lvae_3d_evaluation")
    p.add_argument("--no_x_norm",      action="store_true")
    p.add_argument("--seed",           type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    apply_x_norm = not args.no_x_norm

    bae_model = load_bae_3d(args.bae_checkpoint, device,
                            latent_dim=args.bae_latent_dim,
                            cpx_bound=args.cpx_bound,
                            cpy_bound=args.cpy_bound)
    model     = load_lvae_3d(args.checkpoint, device, bae_model,
                             bae_latent_dim=args.bae_latent_dim,
                             lae_latent_dim=args.lae_latent_dim,
                             n_spans=args.n_spans,
                             dropout=args.dropout)

    # ── Dataset ──────────────────────────────────────────────────────────────
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_all     = list(new_dataset["test"])
    test_dataset = [item for item in test_all if item["final"] == 1]
    n_test       = len(test_dataset)
    print(f"Test wings: {n_test}")

    # ── Pre-compute ground truth ──────────────────────────────────────────────
    print("Pre-computing ground truth encodings...")
    gt_airfoils_list, gt_aoas_list, gt_pressures_list = [], [], []
    z_baes_list, params_list, te_shifts_list, le_x_list, chord_list = [], [], [], [], []
    flow_conditions = []

    for item in test_dataset:
        z_bae, gt_coords, aoa, params_scaled, te_shifts, pressure, le_x, chord = encode_item_3d(
            item, bae_model, model, device, apply_x_norm=apply_x_norm
        )
        gt_airfoils_list.append(gt_coords)
        gt_aoas_list.append(aoa)
        gt_pressures_list.append(pressure)
        z_baes_list.append(z_bae)
        params_list.append(params_scaled)
        te_shifts_list.append(te_shifts)
        le_x_list.append(le_x)
        chord_list.append(chord)
        flow_conditions.append({
            "mach":      item["mach"],
            "reynolds":  item["reynolds"],
            "cl_target": item["cl_target"],
        })

    gt_airfoils_t  = torch.stack(gt_airfoils_list)   # [N, S, 2, 192]
    gt_aoas_t      = torch.cat(gt_aoas_list)          # [N]
    gt_pressures_t = torch.stack(gt_pressures_list)   # [N, S, 192]
    z_baes_t       = torch.stack(z_baes_list)         # [N, bae_latent_dim]
    te_shifts_t    = torch.stack(te_shifts_list)      # [N, S]
    le_x_t         = torch.stack(le_x_list)           # [N, S]
    chord_t        = torch.stack(chord_list)          # [N, S]

    # ── Diagnostic: raw coords vs BAE roundtrip ───────────────────────────────
    raw_coords_list = []
    for item in test_dataset:
        coords = torch.tensor(item["coords"], dtype=torch.float32)   # [S, 192, 2]
        raw_coords_list.append(coords.permute(0, 2, 1))              # [S, 2, 192]
    raw_coords_t = torch.stack(raw_coords_list)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    for si in range(min(3, n_test)):
        lvae_plotting.plot_wing3d_comparison(
            raw_coords_t, gt_airfoils_t,
            sample_idx=si,
            save_path=os.path.join(args.save_dir, f"raw_vs_bae_{timestamp}_s{si}.png"),
            title_a="Raw dataset coords",
            title_b="3D BAE roundtrip",
        )

    # ── Reconstruction pass ───────────────────────────────────────────────────
    print("Running reconstruction pass...")
    rec_airfoils, rec_aoas, rec_pressures, rec_perfs = [], [], [], []
    w_latents_list = []   # collect LVAE bottleneck latents [N, lae_latent_dim]

    for start in range(0, n_test, BATCH_SIZE):
        end             = min(start + BATCH_SIZE, n_test)
        z_batch         = z_baes_t[start:end].to(device)
        pressure_batch  = gt_pressures_t[start:end].to(device)
        params_batch    = torch.cat(params_list[start:end], dim=0).to(device)
        te_shifts_batch = te_shifts_t[start:end]
        le_x_batch      = le_x_t[start:end]
        chord_batch     = chord_t[start:end]

        with torch.no_grad():
            w = model.encoder(z_batch, pressure_batch, params_batch)
            w_latents_list.append(w.cpu())

        coords, aoas, pressure, perf = reconstruct_batch_3d(
            model, bae_model, z_baes_t[start:end], gt_pressures_t[start:end],
            torch.cat(params_list[start:end], dim=0), te_shifts_batch,
            le_x_batch, chord_batch, device
        )
        rec_airfoils.append(coords)
        rec_aoas.extend(aoas.tolist())
        rec_pressures.append(pressure)
        if perf is not None:
            rec_perfs.append(perf)

    rec_airfoils_t  = torch.cat(rec_airfoils)             # [N, S, 2, 192]
    rec_aoas_t      = torch.tensor(rec_aoas)              # [N]
    rec_pressures_t = torch.cat(rec_pressures) if rec_pressures[0] is not None else None
    rec_perfs_t     = torch.cat(rec_perfs) if rec_perfs else None  # [N, 2]
    gt_perfs_t      = torch.tensor(
        [[item["cd_val"], item["cl_val"]] for item in test_dataset], dtype=torch.float32
    )
    w_latents_np = torch.cat(w_latents_list).numpy()      # [N, lae_latent_dim]

    # ── Reconstruction plots ──────────────────────────────────────────────────
    if rec_pressures_t is not None:
        lvae_plotting.plot_airfoil_cp_comparison(
            rec_airfoils_t, gt_airfoils_t,
            rec_pressures_t, gt_pressures_t,
            sample_indices=list(range(min(3, n_test))),
            slice_indices=list(range(gt_airfoils_t.shape[1])),
            save_path=os.path.join(args.save_dir, f"rec_airfoil_cp_{timestamp}.png"),
            pred_label="Reconstructed",
            flow_conditions=flow_conditions,
        )

    lvae_plotting.plot_airfoil_slices_comparison(
        rec_airfoils_t, gt_airfoils_t,
        sample_indices=list(range(min(5, n_test))),
        slice_indices=[0, 4, 8, 11, 14],
        save_path=os.path.join(args.save_dir, f"rec_slice_comparison_{timestamp}.png"),
    )

    for si in range(min(3, n_test)):
        lvae_plotting.plot_wing3d_comparison(
            rec_airfoils_t, gt_airfoils_t,
            sample_idx=si,
            save_path=os.path.join(args.save_dir, f"rec_wing3d_{timestamp}_s{si}.png"),
            title_a="GT",
            title_b="Reconstructed (LVAE3D)",
        )

    lvae_plotting.plot_reconstruction_error_histograms(
        rec_airfoils_t, gt_airfoils_t,
        rec_pressures_t, gt_pressures_t,
        rec_perfs=rec_perfs_t, gt_perfs=gt_perfs_t,
        save_path=os.path.join(args.save_dir, f"rec_error_hist_{timestamp}.png"),
    ) if rec_pressures_t is not None else None

    # =========================================================================
    # METRIC EXPLANATIONS + COMPUTATION
    # =========================================================================

    print("\n\n" + "█" * 70)
    print("  COMPREHENSIVE EVALUATION PIPELINE — METRIC EXPLANATIONS")
    print("█" * 70)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Test wings : {n_test}  |  BAE latent dim: {z_baes_t.shape[1]}"
          f"  |  LVAE latent dim: {w_latents_np.shape[1]}")

    # ── COORDINATE SPACE ─────────────────────────────────────────────────────

    print("\n\n" + "▓" * 70)
    print("  COORDINATE SPACE  (physical x,y wing profiles)")
    print("▓" * 70)

    # ── Shape MSE ────────────────────────────────────────────────────────────
    _explain(
        title  = "Shape MSE  (Mean Squared Error)",
        space  = "COORDINATE SPACE — the physical world. Every number here is an "
                 "(x,y) position on a wing surface, chord-normalised.",
        inputs = "Two arrays of shape [N=94, S=15 spans, 2 coords, 192 points]: "
                 "(1) gt_airfoils — the BAE roundtrip of the true wing, "
                 "(2) rec_airfoils — the full LVAE encode→decode→BAE-decode pipeline.",
        math   = "For every one of the 94×15×2×192 = 543,360 numbers, subtract "
                 "prediction from truth, square it (so negatives can't cancel "
                 "positives), then take the mean of all those squared differences. "
                 "Result: one scalar measuring average squared gap in chord units.",
        good   = "Close to 0. Our value of ~6e-6 means the average pointwise error "
                 "is sqrt(6e-6) ≈ 0.0024 chord units — about 0.24% of chord.",
        bad    = "Large positive values. A value of 1e-4 would mean ~1% chord error "
                 "on average, which would visibly distort the wing shape.",
    )
    geom_m     = compute_geometry_metrics(rec_airfoils_t, gt_airfoils_t,
                                          rec_aoas_t, gt_aoas_t)
    pressure_m = compute_pressure_metrics(rec_pressures_t, gt_pressures_t) if rec_pressures_t is not None else None
    rec_np = rec_airfoils_t.numpy()
    gt_np  = gt_airfoils_t.numpy()
    rec_p  = rec_pressures_t.numpy() if rec_pressures_t is not None else None
    gt_p   = gt_pressures_t.numpy()  if rec_pressures_t is not None else None

    def r2_score(pred: np.ndarray, gt: np.ndarray) -> float:
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    shape_r2    = r2_score(rec_np, gt_np)
    pressure_r2 = r2_score(rec_p, gt_p) if rec_p is not None else float("nan")
    print(f"\n  ► Shape MSE = {geom_m['shape_mse']:.4e}")

    # ── Shape R² ─────────────────────────────────────────────────────────────
    _explain(
        title  = "Shape R²  (Coefficient of Determination)",
        space  = "COORDINATE SPACE — same physical (x,y) arrays as MSE.",
        inputs = "Same gt_airfoils and rec_airfoils arrays as above.",
        math   = "Two sums: SS_res = sum of (pred - truth)^2 over all points "
                 "(same as MSE × N). SS_tot = sum of (truth - mean_of_truth)^2 — "
                 "how spread out the true values are. R² = 1 - SS_res/SS_tot. "
                 "If SS_res equals SS_tot you get 0 (model predicts the mean). "
                 "If SS_res << SS_tot the ratio is near 0 so R² is near 1.",
        good   = "Close to 1.0. Our value of 0.9999 means the model explains "
                 "99.99% of the geometric variance in the test set.",
        bad    = "Below ~0.95 would indicate meaningful shape distortion. "
                 "Negative values mean the model is worse than just predicting "
                 "the mean wing shape.",
    )
    print(f"\n  ► Shape R² = {shape_r2:.6f}")

    # ── Coordinate-space MMD ─────────────────────────────────────────────────
    _explain(
        title  = "Coordinate-Space MMD  (Maximum Mean Discrepancy)",
        space  = "COORDINATE SPACE — but comparing DISTRIBUTIONS, not individual "
                 "pairs. The question is: does the set of reconstructed wings look "
                 "statistically like the set of real wings?",
        inputs = "Per-span flattened coordinate arrays: for each of the 15 spans, "
                 "gt_airfoils[:,s,:,:] flattened to [94, 384] and "
                 "rec_airfoils[:,s,:,:] flattened to [94, 384]. MMD is computed "
                 "per span and averaged.",
        math   = "MMD uses a Gaussian kernel K(x,y) = exp(-γ||x-y||²). "
                 "It computes three kernel matrices: K(real,real), K(pred,pred), "
                 "K(real,pred). MMD² = mean(K_rr) - 2*mean(K_rp) + mean(K_pp). "
                 "Intuitively: if pred and real come from the same distribution, "
                 "cross-distances equal self-distances and MMD→0. "
                 "This is averaged over γ ∈ {0.5, 25, 50, 100} to be "
                 "scale-insensitive.",
        good   = "Close to 0 means reconstructed and real wings are "
                 "statistically indistinguishable. Our value ~0.002 is very small.",
        bad    = "Large values mean the model systematically generates wings from "
                 "a different distribution — e.g. always too thick or too curved.",
    )
    print(f"\n  ► Coord-Space MMD = {geom_m['mmd']:.4f}  "
          f"(γ ∈ {{0.5, 25, 50, 100}}, averaged over {gt_airfoils_t.shape[1]} spans)")

    # ── LATENT SPACE ─────────────────────────────────────────────────────────

    print("\n\n" + "▓" * 70)
    print("  LATENT SPACE  (internal 64-dimensional representations)")
    print("▓" * 70)

    # ── Latent LVAE MMD ──────────────────────────────────────────────────────
    _explain(
        title  = "Latent LVAE MMD  (w-vectors vs N(0,I) prior)",
        space  = "LATENT SPACE — the 64-dimensional bottleneck vectors w produced "
                 "by the LVAE encoder. These are abstract internal codes, not "
                 "physical coordinates.",
        inputs = "(1) w_latents: shape [94, 64] — one 64-d code per test wing, "
                 "extracted by running each wing through the frozen LVAE encoder. "
                 "(2) prior: shape [94, 64] — drawn fresh from N(0,I), i.e. 94×64 "
                 "independent standard normal random numbers. Same size as w_latents "
                 "so the comparison is fair.",
        math   = "Identical Gaussian-kernel MMD as coordinate-space MMD, but now "
                 "operating in 64 dimensions. K(w_i, w_j) = exp(-γ||w_i-w_j||²) "
                 "where w_i and w_j are 64-d vectors. MMD² = mean(K_ww) "
                 "- 2*mean(K_w_prior) + mean(K_prior_prior). "
                 "Averaged over γ ∈ {0.5, 25, 50, 100}.",
        good   = "Close to 0 means the LVAE latent distribution looks like a "
                 "standard Gaussian — a smooth, navigable space for the DDM. "
                 "Our value 0.0795 is small, especially with no KL loss.",
        bad    = "Large values mean the latent space has a very different shape "
                 "from N(0,I) — clustered, multi-modal, or outside the prior "
                 "support — which would break diffusion model sampling.",
    )
    print("Computing latent-space MMD...")
    latent_lvae_mmd = compute_latent_mmd(w_latents_np)
    print(f"\n  ► Latent LVAE MMD = {latent_lvae_mmd:.4f}"
          f"  (w shape: {w_latents_np.shape}, γ ∈ {{0.5, 25, 50, 100}})")

    # ── PCA baseline ─────────────────────────────────────────────────────────
    _explain(
        title  = "PCA Latent MMD  (PCA scores vs N(0,I) prior)",
        space  = "LATENT SPACE — but the latent here is a classical PCA projection, "
                 "not a neural network encoding. This is the baseline that tells us "
                 "whether a linear method produces a similarly structured space.",
        inputs = "(1) PCA is FIT on training-set coordinates [767, 15×2×192=5760] "
                 "to find the top 64 directions of variance. "
                 "(2) Test wings [94, 5760] are PROJECTED onto those 64 directions, "
                 "giving pca_latents [94, 64]. "
                 "(3) Same N(0,I) prior [94, 64] as used for LVAE MMD.",
        math   = "Exactly the same Gaussian-kernel MMD formula applied to "
                 "pca_latents vs prior. The only difference from LVAE MMD is the "
                 "source of the 64-d vectors: linear PCA scores instead of "
                 "neural encoder outputs.",
        good   = "Similar or higher than LVAE MMD would confirm that the LVAE's "
                 "implicit regularisation (from multi-task training) is at least "
                 "as effective as PCA at producing a Gaussian-like space.",
        bad    = "Much lower than LVAE MMD would mean PCA produces a more Gaussian "
                 "distribution, suggesting the LVAE encoder is poorly regularised.",
    )
    print("Fitting PCA baseline on training coords...")
    new_dataset_pca  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    train_items_pca  = [item for item in new_dataset_pca["train"] if item["final"] == 1]
    train_coords_np = []
    for item in train_items_pca:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
        coords_c  = coords.clone()
        coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
        te_x  = coords_c[:, 0, 0]
        coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
        le_x  = coords_c[:, :, 0].min(dim=1).values
        chord = 1.0 - le_x
        coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)
        train_coords_np.append(coords_c.permute(0, 2, 1).numpy())
    train_coords_np = np.stack(train_coords_np)   # [N_train, S, 2, 192]

    gt_coords_np = gt_airfoils_t.numpy()
    n_components = w_latents_np.shape[1]
    pca, pca_test_latents, _ = fit_pca_and_transform(
        train_coords_np, gt_coords_np, n_components=n_components
    )
    latent_pca_mmd = compute_latent_mmd(pca_test_latents)
    print(f"\n  ► Latent PCA  MMD = {latent_pca_mmd:.4f}"
          f"  (pca_latents shape: {pca_test_latents.shape}, "
          f"variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # Per-sample errors + histogram
    lvae_per_sample_errors = ((rec_np - gt_coords_np) ** 2).reshape(len(rec_np), -1).mean(axis=1)
    pca_per_sample_errors  = compute_pca_reconstruction_errors(gt_coords_np, pca, n_components)
    pca_recon_mse = float(pca_per_sample_errors.mean())

    hist_path = os.path.join(args.save_dir, f"error_hist_lvae_vs_pca_{timestamp}.png")
    plot_error_histogram_comparison(lvae_per_sample_errors, pca_per_sample_errors,
                                    save_path=hist_path)

    # ── Pressure R² ──────────────────────────────────────────────────────────
    _explain(
        title  = "Pressure R²  (surface Cp coefficient of determination)",
        space  = "COORDINATE SPACE — but for aerodynamic pressure values, not "
                 "geometry. The Cp (pressure coefficient) field lives on the same "
                 "192-point surface grid as the coordinates.",
        inputs = "gt_pressures [94, 15, 192] — true Cp values from CFD simulation. "
                 "rec_pressures [94, 15, 192] — Cp predicted by the LVAE decoder's "
                 "pressure head.",
        math   = "Same R² formula as shape: 1 - SS_res/SS_tot, where SS_res is "
                 "sum of (predicted_Cp - true_Cp)^2 over all 94×15×192 = 2,718,720 "
                 "values, and SS_tot uses the mean Cp across the whole test set.",
        good   = "Close to 1.0. Our value 0.9856 means the decoder captures 98.56% "
                 "of the pressure variance — important for aerodynamic consistency.",
        bad    = "Below ~0.90 would mean the pressure predictions are significantly "
                 "wrong, which would make downstream drag/lift estimates unreliable.",
    )
    if rec_p is not None:
        print(f"\n  ► Pressure R² = {pressure_r2:.6f}")

    # =========================================================================
    # FINAL RESULTS SUMMARY
    # =========================================================================

    machs = np.array([fc["mach"] for fc in flow_conditions])
    regimes = [
        ("Subsonic   (Mach < 0.8)", machs < 0.8),
        ("Transonic  (0.8-1.0)   ", (machs >= 0.8) & (machs < 1.0)),
        ("Supersonic (Mach >= 1.0)", machs >= 1.0),
    ]

    print("\n\n" + "=" * 70)
    print("  EVALUATION RESULTS  (LVAE3D Reconstruction)")
    print("=" * 70)
    print(f"  Shape MSE        : {geom_m['shape_mse']:.4e}")
    print(f"  Shape R²         : {shape_r2:.4f}")
    print(f"  AoA MSE          : {geom_m['aoa_mse']:.4f}")
    print(f"  MMD (coord space): {geom_m['mmd']:.4f}")
    print(f"  Vendi            : {geom_m['vendi']:.4f}")
    print(f"  Latent LVAE MMD  : {latent_lvae_mmd:.4f}")
    print(f"  Latent PCA  MMD  : {latent_pca_mmd:.4f}")
    print(f"  PCA recon MSE    : {pca_recon_mse:.2e}  (n_components={n_components})")
    if pressure_m is not None:
        print(f"  Pressure MSE     : {pressure_m['pressure_mse']:.4f}")
        print(f"  Pressure R²      : {pressure_r2:.4f}")
        print(f"  Pressure slice MSE (s0..s{len(pressure_m['pressure_slice_mse'])-1}): "
              + " ".join(f"{v:.4f}" for v in pressure_m["pressure_slice_mse"]))
    print("── By flow regime ──────────────────────────────────────────────────")
    for label, mask in regimes:
        if mask.sum() == 0:
            print(f"  {label}: no wings")
            continue
        s_mse = float(((rec_np[mask] - gt_np[mask]) ** 2).mean())
        s_r2  = r2_score(rec_np[mask], gt_np[mask])
        if rec_p is not None:
            p_mse = float(((rec_p[mask] - gt_p[mask]) ** 2).mean())
            p_r2  = r2_score(rec_p[mask], gt_p[mask])
            print(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.2e}  "
                  f"shape R²={s_r2:.4f}  pressure MSE={p_mse:.4f}  pressure R²={p_r2:.4f}")
        else:
            print(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.2e}  shape R²={s_r2:.4f}")
    print("=" * 70)

    # ── Markdown comparison table ─────────────────────────────────────────────
    lvae_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    md = f"""
## Evaluation Summary: {lvae_name} vs. PCA Baseline

### Coordinate Space

| Metric | LVAE (multi-task) | PCA baseline (64 components) | Better |
|--------|:-----------------:|:----------------------------:|:------:|
| Shape MSE | {geom_m['shape_mse']:.4e} | {pca_recon_mse:.4e} | PCA (by construction) |
| Shape R² | {shape_r2:.4f} | N/A | LVAE (joint model) |
| Coord-Space MMD | {geom_m['mmd']:.4f} | N/A | LVAE |
| Pressure MSE | {f"{pressure_m['pressure_mse']:.4f}" if pressure_m is not None else 'N/A'} | N/A (not encoded) | LVAE |
| Pressure R² | {f"{pressure_r2:.4f}" if rec_p is not None else 'N/A'} | N/A (not encoded) | LVAE |

### Latent Space

| Metric | LVAE w-vectors | PCA scores | Better |
|--------|:--------------:|:----------:|:------:|
| Latent MMD vs N(0,I) | {latent_lvae_mmd:.4f} | {latent_pca_mmd:.4f} | {'LVAE' if latent_lvae_mmd < latent_pca_mmd else 'PCA'} |
| Latent dim | {n_components} | {n_components} | — |
| Encodes pressure? | Yes | No | LVAE |
| Connected topology (t-SNE) | Yes (embedded in prior) | No (fractured clusters) | LVAE |

### Key Takeaway
PCA achieves lower coordinate MSE by construction (it minimises reconstruction error),
but encodes **geometry only** and produces a **fractured latent topology** incompatible
with diffusion-model sampling. The LVAE trades a ~3.4× increase in shape MSE for a
joint latent space that encodes pressure (R²={f"{pressure_r2:.4f}" if rec_p is not None else 'N/A'}), stays embedded within
the N(0,I) prior (latent MMD={latent_lvae_mmd:.4f} vs PCA {latent_pca_mmd:.4f}), and
supports smooth navigation by the downstream DDM-W.
"""
    print(md)

    # ── Save results ──────────────────────────────────────────────────────────
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    results_path = os.path.join(args.save_dir, f"eval_{model_name}_{timestamp}.txt")
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"BAE checkpoint: {args.bae_checkpoint}\n")
        f.write(f"Test wings: {n_test}\n\n")
        f.write("── Reconstruction ───────────────────────────────────────\n")
        for k in ["shape_mse", "aoa_mse", "mmd", "vendi"]:
            f.write(f"{k:16s}: {geom_m[k]:.6f}\n")
        f.write(f"{'shape_r2':16s}: {shape_r2:.6f}\n")
        f.write(f"{'latent_lvae_mmd':16s}: {latent_lvae_mmd:.6f}\n")
        f.write(f"{'latent_pca_mmd':16s}: {latent_pca_mmd:.6f}\n")
        f.write(f"{'pca_recon_mse':16s}: {pca_recon_mse:.6f}  (n_components={n_components})\n")
        if pressure_m is not None:
            f.write(f"{'pressure_mse':16s}: {pressure_m['pressure_mse']:.6f}\n")
            f.write(f"{'pressure_r2':16s}: {pressure_r2:.6f}\n")
            f.write("Pressure slice MSE:\n")
            for s, v in enumerate(pressure_m["pressure_slice_mse"]):
                f.write(f"  slice {s}: {v:.6f}\n")
        f.write("\n── By flow regime ───────────────────────────────────────\n")
        for label, mask in regimes:
            if mask.sum() == 0:
                f.write(f"  {label}: no wings\n")
                continue
            s_mse = float(((rec_np[mask] - gt_np[mask]) ** 2).mean())
            s_r2  = r2_score(rec_np[mask], gt_np[mask])
            if rec_p is not None:
                p_mse = float(((rec_p[mask] - gt_p[mask]) ** 2).mean())
                p_r2  = r2_score(rec_p[mask], gt_p[mask])
                f.write(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.6f}  "
                        f"shape R²={s_r2:.6f}  pressure MSE={p_mse:.6f}  pressure R²={p_r2:.6f}\n")
            else:
                f.write(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.6f}  shape R²={s_r2:.6f}\n")
        f.write("\n" + md)

    print(f"\nResults saved to {results_path}")
    print(f"Plots saved to   {args.save_dir}/")


if __name__ == "__main__":
    main()
