"""
Standalone script: load lvae_3d_v8_best.pth, encode train+test sets,
plot per-dimension std of the 64-dim LVAE latent space.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

LVAE_CHECKPOINT = "results/lvae_3d/lvae_3d_v8_best.pth"
BAE_CHECKPOINT  = "results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt"
SLICES_PKL      = "Wing_TL/data/processed/new_dataset_slices.pkl"
SCALARS_PKL     = "Wing_TL/data/processed/new_dataset_scalars.pkl"
SAVE_DIR        = "results/lvae_evaluation"
BATCH_SIZE      = 32
DEVICE          = "cpu"


def load_models():
    from engiopt.lvae.train_lvae_3d import (
        LAEEncoderJoint3D, LAEDecoder3D, LVAE3D, load_bae_3d,
    )
    from engiopt.data_processing.utils import scaler as make_scaler

    bae_model = load_bae_3d(BAE_CHECKPOINT, DEVICE)

    ckpt = torch.load(LVAE_CHECKPOINT, map_location=DEVICE, weights_only=False)

    lae_latent_dim    = ckpt["lae_latent_dim"]
    bae_latent_dim    = ckpt["bae_latent_dim"]
    n_spans           = ckpt["n_spans"]
    dropout           = ckpt.get("dropout", 0.0)
    pressure_length   = ckpt.get("pressure_length", 192)
    pressure_embed_dim = ckpt.get("pressure_embed_dim", 16)

    encoder = LAEEncoderJoint3D(
        bae_latent_dim=bae_latent_dim,
        c_dim=len(ckpt["params_mean_std"][0]),
        n_spans=n_spans,
        pressure_length=pressure_length,
        pressure_embed_dim=pressure_embed_dim,
        lae_latent_dim=lae_latent_dim,
        dropout=dropout,
    )
    encoder.load_state_dict(ckpt["encoder"])

    # Infer decoder output dims from state dict
    dec_hidden = 512  # last backbone hidden dim
    decoder = LAEDecoder3D(
        bae_latent_dim=bae_latent_dim,
        c_dim=len(ckpt["params_mean_std"][0]),
        lae_latent_dim=lae_latent_dim,
        n_spans=n_spans,
        pressure_length=pressure_length,
        dropout=dropout,
    )
    decoder.load_state_dict(ckpt["decoder"])

    lvae = LVAE3D(
        encoder=encoder,
        decoder=decoder,
        bae_model=bae_model,
        lae_latent_dim=lae_latent_dim,
        params_mean_std=ckpt.get("params_mean_std"),
        aoas_mean_std=ckpt.get("aoas_mean_std"),
        pressures_mean_std=ckpt.get("pressures_mean_std"),
    )
    if "active_latent_mask" in ckpt:
        lvae.active_latent_mask = ckpt["active_latent_mask"].to(DEVICE)
    lvae.eval()

    return lvae, bae_model


def encode_item_bae(item, bae_model):
    """Preprocess coords and encode through 3D BAE → [bae_latent_dim]."""
    coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
    te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]
    coords_c  = coords.clone()
    coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
    te_x  = coords_c[:, 0, 0]
    coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
    le_x  = coords_c[:, :, 0].min(dim=1).values
    chord = 1.0 - le_x
    coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)
    x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(DEVICE)   # [1, S, 2, 192]
    with torch.no_grad():
        z_bae = bae_model.encode(x_wing).squeeze(0).cpu()         # [bae_latent_dim]
    return z_bae


def collect_latents(lvae, bae_model, items):
    all_w = []
    for start in range(0, len(items), BATCH_SIZE):
        chunk = items[start:start + BATCH_SIZE]
        z_bae_b, params_b, press_b = [], [], []

        for item in chunk:
            z_bae_b.append(encode_item_bae(item, bae_model))

            flow_p = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            c_dim  = len(lvae.scaler_params.mean)
            params = torch.tensor(flow_p[:c_dim], dtype=torch.float32)
            params_b.append(lvae.scaler_params.transform(params.unsqueeze(0)).squeeze(0))

            press_b.append(torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32))

        z_bae_bt  = torch.stack(z_bae_b).to(DEVICE)          # [B, bae_latent_dim]
        params_bt = torch.stack(params_b).to(DEVICE)          # [B, c_dim]
        press_bt  = torch.stack(press_b).to(DEVICE)           # [B, n_spans, 192]

        with torch.no_grad():
            w = lvae.encoder(z_bae_bt, press_bt, params_bt)   # [B, lae_latent_dim]
        all_w.append(w.cpu())

        print(f"  Encoded {min(start + BATCH_SIZE, len(items))}/{len(items)}...")

    return torch.cat(all_w, dim=0).numpy()


def main():
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    from engiopt.lvae import plotting as lvae_plotting

    print("Loading models...")
    lvae, bae_model = load_models()
    print(f"  LVAE latent dim: {lvae.lae_latent_dim}")
    active_mask = lvae.active_latent_mask.cpu().numpy()
    print(f"  Active dims: {active_mask.sum()}/{lvae.lae_latent_dim}")

    dataset = NewWingsDataset(SLICES_PKL, SCALARS_PKL, seed=0)
    train_items = [item for item in dataset["train"] if item["final"] == 1]
    test_items  = [item for item in dataset["test"]  if item["final"] == 1]
    print(f"  Train: {len(train_items)} wings, Test: {len(test_items)} wings")

    print("Encoding train set...")
    train_z = collect_latents(lvae, bae_model, train_items)
    print("Encoding test set...")
    test_z  = collect_latents(lvae, bae_model, test_items)

    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, f"latent_std_lvae3d_v8_{ts}.png")

    lvae_plotting.plot_latent_std(
        train_z, test_z,
        active_mask=active_mask,
        threshold=0.02,
        save_path=save_path,
    )
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
