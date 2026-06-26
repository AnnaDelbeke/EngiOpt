"""
Generate a wing from DDM-W and export it as a binary STL file for 3D printing.

The wing is lofted from 15 spanwise cross-sections:
  - x (chordwise) = x/c * chord_m
  - y (thickness) = y/c * chord_m
  - z (spanwise)  = eta in metres (real physical coordinate from the dataset)

Quad strips between adjacent slices are triangulated; root and tip end-caps
are fan-triangulated from their centroids so the mesh is watertight.

Usage
-----
    source .venv/bin/activate
    python export_wing_stl.py                     # first test wing, generated
    python export_wing_stl.py --case_idx 3        # specific test wing
    python export_wing_stl.py --chord_m 0.5       # scale chord to 0.5 m
    python export_wing_stl.py --use_gt            # export ground truth instead
    python export_wing_stl.py --out my_wing.stl
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from engiopt.data_processing.utils import scaler
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm.ddm_w.train_ddm_w_3d import Config, build_sampler, load_lvae_3d
from engiopt.lvae.evaluate_lvae_3d import load_bae_3d

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"
BAE_CKP      = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
LVAE_CKP     = "results/lvae_3d/lvae_3d_v29_best.pth"
DDM_CKP      = "results/ddm_w_3d/ddm_w_3d_v29_best.pth"


# ---------------------------------------------------------------------------
# STL writer
# ---------------------------------------------------------------------------

def write_stl_binary(path: str, triangles: np.ndarray):
    """Write triangles [N, 3, 3] (float32 xyz) as a binary STL."""
    n = len(triangles)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        for tri in triangles:
            v0, v1, v2 = tri.astype(np.float32)
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            normal = (normal / norm) if norm > 1e-12 else normal
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def loft_to_triangles(vertices: np.ndarray) -> np.ndarray:
    """
    Loft S cross-sections into a closed triangle mesh.

    vertices : [S, P, 3]
    returns  : [N_tri, 3, 3]
    """
    S, P, _ = vertices.shape
    tris = []

    # Outer surface
    for s in range(S - 1):
        for p in range(P):
            p1 = (p + 1) % P
            v00 = vertices[s,     p]
            v10 = vertices[s + 1, p]
            v01 = vertices[s,     p1]
            v11 = vertices[s + 1, p1]
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    # End caps (fan from centroid)
    for s, inward in [(0, True), (S - 1, False)]:
        ring   = vertices[s]
        centre = ring.mean(axis=0)
        for p in range(P):
            p1 = (p + 1) % P
            if inward:
                tris.append([centre, ring[p1], ring[p]])
            else:
                tris.append([centre, ring[p],  ring[p1]])

    return np.array(tris, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ddm_w_model(device):
    cfg = Config()
    cfg.lvae_checkpoint = LVAE_CKP
    bae_model  = load_bae_3d(BAE_CKP, device, latent_dim=cfg.bae_latent_dim)
    lvae_model = load_lvae_3d(cfg, bae_model)
    lvae_model.encoder.to(device)

    ckpt    = torch.load(DDM_CKP, map_location="cpu", weights_only=False)
    sampler = build_sampler(cfg)
    w_dim   = cfg.lae_latent_dim
    saved   = ckpt["denoiser"]
    if isinstance(saved, MLPDenoiser):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=w_dim, c_dim=cfg.c_dim)
        denoiser.load_state_dict(saved)

    ddm_w = DDM_W3D(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=w_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get("w_pressure", 1.0),
        w_aoa=ckpt.get("w_aoa", 1.0),
        lvae_params_dim=cfg.c_dim,
        params_mean_std=ckpt.get("params_mean_std"),
        aoas_mean_std=ckpt.get("aoas_mean_std"),
    )
    ddm_w.w_mean     = ckpt.get("w_mean")
    ddm_w.w_std      = ckpt.get("w_std")
    ddm_w.p_ddm_mean = ckpt.get("p_ddm_mean", 0.0)
    ddm_w.p_ddm_std  = ckpt.get("p_ddm_std",  1.0)
    ddm_w.denoiser.to(device)
    return bae_model, lvae_model, ddm_w


# ---------------------------------------------------------------------------
# Encode one item → w_init, params_norm
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_item(item, bae_model, lvae_model, ddm_w, device):
    coords    = torch.tensor(item["coords"],        dtype=torch.float32)
    te_shifts = torch.tensor(item["te_shifts"],     dtype=torch.float32)
    pressure  = torch.tensor(item["coef_pressure"], dtype=torch.float32)

    # Apply the same normalisation as training
    coords_c = coords.clone()
    coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
    te_x = coords_c[:, 0, 0]
    coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
    le_x  = coords_c[:, :, 0].min(dim=1).values
    chord = 1.0 - le_x
    coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

    x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)   # [1, S, 2, 192]
    z_bae  = bae_model.encode(x_wing)                             # [1, bae_latent_dim]

    flow = np.array([[item["mach"], item["reynolds"],
                      item["cl_target"], item["area_case_ratio"]]], dtype=np.float32)
    lvae_ps = getattr(lvae_model, "scaler_params", None)
    flow_lvae = torch.tensor(
        lvae_ps.transform(flow) if lvae_ps is not None else flow,
        dtype=torch.float32, device=device,
    )
    pres_d = pressure.unsqueeze(0).to(device)
    w_gt   = lvae_model.encoder(z_bae, pres_d, flow_lvae)         # [1, w_dim]

    # Normalise w for DDM conditioning
    if ddm_w.w_mean is not None:
        w_mean = ddm_w.w_mean.to(device)
        w_std  = ddm_w.w_std.to(device)
        w_init = (w_gt - w_mean) / w_std
    else:
        w_init = w_gt

    # Normalise params for DDM conditioning
    pms = ddm_w.scaler_params if hasattr(ddm_w, "scaler_params") else None
    if pms is not None:
        params_norm = torch.tensor(pms.transform(flow), dtype=torch.float32, device=device)
    else:
        params_norm = torch.tensor(flow, dtype=torch.float32, device=device)

    return w_init, params_norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_idx", type=int,   default=0,
                   help="Index into the test set (0-based)")
    p.add_argument("--chord_m",  type=float, default=1.0,
                   help="Reference chord in metres for scaling x/c and y/c")
    p.add_argument("--out",      type=str,   default="wing.stl")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--use_gt",   action="store_true",
                   help="Export ground-truth geometry instead of generated")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bae_model, lvae_model, ddm_w = load_ddm_w_model(device)

    dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    test_set = [it for it in dataset["test"] if it["final"] == 1]
    item     = test_set[args.case_idx]
    print(f"Wing: case {int(item['case_num'])},  Mach={item['mach']:.3f},  "
          f"Re={item['reynolds']:.2e},  CL_target={item['cl_target']:.3f}")

    etas = np.array(item["transforms"], dtype=np.float32)   # [S] spanwise metres

    if args.use_gt:
        # Ground truth: coords are [S, 192, 2] x/c, y/c
        coords = np.array(item["coords"], dtype=np.float32)  # [S, 192, 2]
        print("Using ground-truth geometry.")
    else:
        w_init, params_norm = encode_item(item, bae_model, lvae_model, ddm_w, device)
        with torch.no_grad():
            gen_coords, *_ = ddm_w.generate(w_init, params_norm, device)
        # gen_coords: [1, S, 2, 192] → [S, 192, 2]
        coords = gen_coords.squeeze(0).permute(0, 2, 1).cpu().numpy()
        print("Generated geometry.")

    S, P, _ = coords.shape
    vertices = np.zeros((S, P, 3), dtype=np.float32)
    vertices[:, :, 0] = coords[:, :, 0] * args.chord_m   # x chordwise
    vertices[:, :, 1] = coords[:, :, 1] * args.chord_m   # y thickness
    vertices[:, :, 2] = etas[:, np.newaxis]               # z spanwise (metres)

    triangles = loft_to_triangles(vertices)
    write_stl_binary(args.out, triangles)
    print(f"Saved: {args.out}  ({len(triangles):,} triangles, "
          f"{S} slices × {P} points, chord={args.chord_m} m, "
          f"span={etas[-1]:.3f} m)")


if __name__ == "__main__":
    main()
