"""
Training script for DDM_W using the 3D BAE + LVAE3D pipeline.

Differences from train_ddm_w.py:
  - Loads BezierAutoencoder3D (joint wing latent [B, bae_latent_dim])
  - Loads LVAE3D (encoder/decoder operate on joint latents)
  - precompute_w encodes full wing at once (no per-slice loop)
  - Saves to results/ddm_w_3d/ — never touches existing DDM_W checkpoints

Usage
-----
    python -m engiopt.ddm.ddm_w.train_ddm_w_3d \
        --bae_checkpoint  results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --lvae_checkpoint results/lvae_3d/lvae_3d_v1_best.pth \
        --model_name ddm_w_3d_v1 \
        --wandb
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.lvae.train_lvae_3d import LVAE3D, LAEEncoder3D, LAEEncoderJoint3D, LAEDecoder3D, load_bae_3d
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm.ddm_w.ddm_w_3d import DDM_W3D
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE 3D
    bae_checkpoint:  str = ""
    bae_latent_dim:  int = 256
    n_spans:         int = 9

    # LVAE 3D
    lvae_checkpoint: str = ""
    lae_latent_dim:  int = 64
    c_dim:           int = 4

    # DDM_W (same denoiser as before)
    w_dim: int         = 64
    hidden_dims: tuple = (512, 512, 512, 512)
    t_embed_dim: int   = 128
    dropout: float     = 0.0
    w_pressure: float  = 1.0
    w_aoa: float       = 1.0

    # Diffusion
    num_diffusion_steps: int = 1000
    cosine_schedule: bool    = False

    # Training
    batch_size: int  = 64
    lr: float        = 1e-4
    n_epochs: int    = 20000
    grad_clip: float = 1.0
    save_every: int  = 200

    # Outputs
    save_dir:   str = "results/ddm_w_3d"
    model_name: str = "ddm_w_3d_v1"
    seed: int       = 0
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WDataset(Dataset):
    def __init__(self, w_opts, aoas, params, w_inits, pressures,
                 scaler_params=None, scaler_aoas=None):
        self.w_opts    = w_opts
        self.aoas      = aoas
        self.params    = params
        self.w_inits   = w_inits
        self.pressures = pressures
        self.scaler_params = scaler_params
        self.scaler_aoas   = scaler_aoas

    def __len__(self): return len(self.w_opts)

    def __getitem__(self, idx):
        w_opt    = self.w_opts[idx].clone()
        aoa      = self.aoas[idx].clone()
        params   = self.params[idx].clone()
        w_init   = self.w_inits[idx].clone()
        pressure = self.pressures[idx].clone()
        if self.scaler_params is not None: params = self.scaler_params.transform(params)
        if self.scaler_aoas   is not None: aoa    = self.scaler_aoas.transform(aoa)
        return w_opt, aoa, params, w_init, pressure


# ---------------------------------------------------------------------------
# Pre-computation: 3D BAE → LVAE3D encoder → w
# ---------------------------------------------------------------------------

def precompute_w(dataset_items, initial_by_case, bae_model, lvae_model,
                 device, c_dim=4):
    """Encode each wing: full wing coords → 3D BAE latent → LVAE3D encoder → w."""
    print(f"Pre-computing w encodings for {len(dataset_items)} samples...")

    w_opts_list, w_inits_list, aoas_list, params_list, pressures_list = [], [], [], [], []

    bae_model.eval()
    lvae_model.encoder.eval()

    with torch.no_grad():
        for i, item in enumerate(dataset_items):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(dataset_items)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            # Encode full wing at once: [1, S, 2, 192]
            x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)
            z_bae  = bae_model.encode(x_wing)                              # [1, bae_latent_dim]

            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_t = torch.tensor(flow, dtype=torch.float32)
            params_list.append(flow_t)

            # Normalise params for LVAE
            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            lvae_ps = getattr(lvae_model, 'scaler_params', None)
            if lvae_ps is not None:
                flow_lvae_np = lvae_ps.transform(flow_np)
            else:
                flow_lvae_np = flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)

            lvae_model.encoder.to(device)
            pressure = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)
            is_joint = isinstance(lvae_model.encoder, LAEEncoderJoint3D)
            if is_joint:
                pressure_dev = pressure.unsqueeze(0).to(device)            # [1, S, 192]
                w = lvae_model.encoder(z_bae, pressure_dev, flow_lvae)     # [1, w_dim]
            else:
                w = lvae_model.encoder(z_bae, pressure.unsqueeze(0).to(device), flow_lvae)  # [1, w_dim]
            w_opts_list.append(w.squeeze(0).cpu())

            # w_init: encode the initial wing
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init    = initial_by_case[case_num]
                ic      = torch.tensor(init["coords"],    dtype=torch.float32)
                it      = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ic_c    = ic.clone()
                ic_c[:, :, 1] -= it.unsqueeze(1)
                te_x_i  = ic_c[:, 0, 0]
                ic_c[:, :, 0] += (1.0 - te_x_i).unsqueeze(1)
                le_x_i  = ic_c[:, :, 0].min(dim=1).values
                chord_i = 1.0 - le_x_i
                ic_c[:, :, 0] = (ic_c[:, :, 0] - le_x_i.unsqueeze(1)) / chord_i.unsqueeze(1)
                x_init  = ic_c.permute(0, 2, 1).unsqueeze(0).to(device)
                z_init  = bae_model.encode(x_init)                         # [1, bae_latent_dim]
                init_pressure = torch.tensor(np.array(init["coef_pressure"]), dtype=torch.float32)
            else:
                z_init = z_bae
                init_pressure = pressure

            if is_joint:
                init_pressure_dev = init_pressure.unsqueeze(0).to(device)
                w_init = lvae_model.encoder(z_init, init_pressure_dev, flow_lvae)
            else:
                w_init = lvae_model.encoder(z_init, init_pressure.unsqueeze(0).to(device), flow_lvae)  # [1, w_dim]
            w_inits_list.append(w_init.squeeze(0).cpu())

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            aoas_list.append(aoa.unsqueeze(0) if aoa.ndim == 0 else aoa)

            pressures_list.append(pressure)

    print("Pre-computation complete.")
    return (
        torch.stack(w_opts_list),
        torch.stack(aoas_list),
        torch.stack(params_list),
        torch.stack(w_inits_list),
        torch.stack(pressures_list),
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_lvae_3d(cfg: Config, bae_model: BezierAutoencoder3D) -> LVAE3D:
    ckpt = torch.load(cfg.lvae_checkpoint, map_location=cfg.device, weights_only=False)

    # prefer checkpoint-saved dims over config defaults
    n_spans        = ckpt.get("n_spans",        cfg.n_spans)
    lae_latent_dim = ckpt.get("lae_latent_dim", cfg.lae_latent_dim)
    bae_latent_dim = ckpt.get("bae_latent_dim", cfg.bae_latent_dim)

    # infer dropout from key spacing: Linear+GELU+Dropout → spacing 3; Linear+GELU → spacing 2
    enc_keys  = sorted(ckpt.get("encoder", {}).keys())
    lin_idxs  = [int(k.split(".")[1]) for k in enc_keys if k.endswith(".weight")]
    spacing   = (lin_idxs[1] - lin_idxs[0]) if len(lin_idxs) >= 2 else 2
    dropout   = 0.1 if spacing == 3 else 0.0  # value only needs to be > 0 to add Dropout layer

    decoder_state      = ckpt.get("decoder", {})
    has_perf_head      = "perf_head.weight" in decoder_state or "perf_head.weight_orig" in decoder_state
    has_sn             = any("weight_orig" in k for k in decoder_state)
    joint_encoder      = ckpt.get("joint_encoder", False)
    pressure_embed_dim = ckpt.get("pressure_embed_dim", 16)

    if joint_encoder:
        encoder = LAEEncoderJoint3D(
            bae_latent_dim=bae_latent_dim, c_dim=cfg.c_dim,
            n_spans=n_spans, pressure_embed_dim=pressure_embed_dim,
            lae_latent_dim=lae_latent_dim, dropout=dropout,
        ).to(cfg.device)
    else:
        encoder = LAEEncoder3D(
            bae_latent_dim=bae_latent_dim, c_dim=cfg.c_dim,
            lae_latent_dim=lae_latent_dim, dropout=dropout,
        ).to(cfg.device)
    decoder = LAEDecoder3D(
        bae_latent_dim=bae_latent_dim, c_dim=cfg.c_dim,
        lae_latent_dim=lae_latent_dim, n_spans=n_spans,
        dropout=dropout, use_perf_head=has_perf_head,
        spectral_norm=has_sn,
    ).to(cfg.device)

    params_mean_std    = ckpt.get("params_mean_std")
    aoas_mean_std      = ckpt.get("aoas_mean_std")
    pressures_mean_std = ckpt.get("pressures_mean_std")

    lvae = LVAE3D(
        encoder=encoder, decoder=decoder, bae_model=bae_model,
        lae_latent_dim=lae_latent_dim,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        pressures_mean_std=pressures_mean_std,
    ).to(cfg.device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    if "perfs_mean_std" in ckpt and ckpt["perfs_mean_std"] is not None:
        from engiopt.data_processing.utils import scaler as make_scaler
        lvae.scaler_perfs = make_scaler(ckpt["perfs_mean_std"])
    if "active_latent_mask" in ckpt:
        lvae.active_latent_mask = ckpt["active_latent_mask"].to(cfg.device)

    # update config dims to match what was loaded
    cfg.lae_latent_dim = lae_latent_dim
    cfg.bae_latent_dim = bae_latent_dim
    cfg.n_spans        = n_spans
    cfg.w_dim          = lae_latent_dim

    lvae.encoder.eval(); lvae.decoder.eval()
    for p in list(lvae.encoder.parameters()) + list(lvae.decoder.parameters()):
        p.requires_grad_(False)
    print(f"Loaded and froze LVAE3D (n_spans={n_spans}, lae_dim={lae_latent_dim}, "
          f"perf_head={has_perf_head}) from {cfg.lvae_checkpoint}")
    return lvae


def build_sampler(cfg: Config):
    return samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4, end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
        cosine=cfg.cosine_schedule,
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model: DDM_W3D, loader: DataLoader, device: str, grad_clip: float):
    model.denoiser.train()
    total_loss = total_w = total_aoa = total_p = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        model.optimizer.zero_grad()
        loss, lw, laoa, lp = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), grad_clip)
        model.optimizer.step()
        total_loss += loss.item(); total_w += lw.item()
        total_aoa  += laoa.item(); total_p  += lp.item()
        n += 1
    d = max(n, 1)
    return total_loss / d, total_w / d, total_aoa / d, total_p / d


@torch.no_grad()
def eval_one_epoch(model: DDM_W3D, loader: DataLoader, device: str):
    model.denoiser.eval()
    total_loss = total_w = total_aoa = total_p = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        loss, lw, laoa, lp = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item(); total_w += lw.item()
        total_aoa  += laoa.item(); total_p  += lp.item()
        n += 1
    model.denoiser.train()
    d = max(n, 1)
    return total_loss / d, total_w / d, total_aoa / d, total_p / d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bae_checkpoint",  type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--model_name",  type=str, default=None)
    p.add_argument("--n_epochs",    type=int, default=None)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--w_pressure",  type=float, default=1.0)
    p.add_argument("--w_aoa",       type=float, default=9.0)
    p.add_argument("--bae_latent_dim", type=int, default=256)
    p.add_argument("--lae_latent_dim", type=int, default=64)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--n_samples",   type=int, default=None,
                   help="Ablation: number of training samples to use (default: all)")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--wandb_project", type=str, default="engiopt-ddm-w-3d")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.bae_checkpoint  = args.bae_checkpoint
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.bae_latent_dim  = args.bae_latent_dim
    cfg.lae_latent_dim  = args.lae_latent_dim
    cfg.lr              = args.lr
    cfg.grad_clip       = args.grad_clip
    cfg.w_pressure      = args.w_pressure
    cfg.w_aoa           = args.w_aoa
    cfg.seed            = args.seed
    if args.model_name  is not None: cfg.model_name = args.model_name
    if args.n_epochs    is not None: cfg.n_epochs   = args.n_epochs
    if args.hidden_dims is not None: cfg.hidden_dims = tuple(args.hidden_dims)

    if args.n_samples is not None:
        cfg.model_name = f"ddm_w_3d_ablation_n{args.n_samples}_s{args.seed}"
        cfg.save_dir   = f"results/ddm_w_3d_ablation/n{args.n_samples}_s{args.seed}"

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Device: {cfg.device}  Model: {cfg.model_name}")

    # 1. Frozen models
    bae_model  = load_bae_3d(cfg.bae_checkpoint, cfg.device,
                              latent_dim=cfg.bae_latent_dim)
    lvae_model = load_lvae_3d(cfg, bae_model)

    lvae_params_scaler = getattr(lvae_model, 'scaler_params', None)

    # 2. Dataset
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train       = list(new_dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset    = [item for item in all_train if item["final"] == 1]
    all_val             = list(new_dataset["val"])
    val_initial_by_case = {item["case_num"]: item for item in all_val if item["initial"] == 1}
    val_dataset         = [item for item in all_val if item["final"] == 1]
    if args.n_samples is not None:
        rng = np.random.default_rng(cfg.seed)
        idxs = rng.choice(len(base_dataset), size=min(args.n_samples, len(base_dataset)), replace=False)
        base_dataset = [base_dataset[i] for i in idxs]
        print(f"Ablation: using {len(base_dataset)} of {len(all_train)} training samples (seed={cfg.seed})")
    print(f"Train: {len(base_dataset)}  Val: {len(val_dataset)}")

    # 3. Condition normalisation
    all_params_np = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas_np = np.array([float(item["alpha"]) for item in base_dataset])
    params_mean_std = (all_params_np.mean(0), all_params_np.std(0))
    aoas_mean_std   = (float(all_aoas_np.mean()), float(all_aoas_np.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    # 4. Pre-compute w encodings
    w_opts, aoas, params_all, w_inits, pressures = precompute_w(
        base_dataset, initial_by_case, bae_model, lvae_model, cfg.device
    )
    w_mean = w_opts.mean(dim=0, keepdim=True)
    w_std  = w_opts.std(dim=0,  keepdim=True).clamp(min=1e-8)
    w_opts_n  = (w_opts  - w_mean) / w_std
    w_inits_n = (w_inits - w_mean) / w_std

    p_mean = float(pressures.mean())
    p_std  = float(pressures.std())
    pressures_n = (pressures - p_mean) / max(p_std, 1e-8)

    dataset = WDataset(w_opts_n, aoas, params_all, w_inits_n, pressures_n,
                       scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers)

    # Val
    val_w, val_aoas, val_params, val_w_inits, val_pressures = precompute_w(
        val_dataset, val_initial_by_case, bae_model, lvae_model, cfg.device
    )
    val_w_n       = (val_w       - w_mean) / w_std
    val_w_inits_n = (val_w_inits - w_mean) / w_std
    val_pressures_n = (val_pressures - p_mean) / max(p_std, 1e-8)
    val_ds = WDataset(val_w_n, val_aoas, val_params, val_w_inits_n, val_pressures_n,
                      scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    # 5. Build DDM_W (same denoiser — w_dim is LVAE latent, unchanged)
    denoiser = MLPDenoiser(
        w_dim=cfg.w_dim, c_dim=cfg.c_dim,
        hidden_dims=cfg.hidden_dims, t_embed_dim=cfg.t_embed_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)

    sampler = build_sampler(cfg)

    ddm_w = DDM_W3D(
        denoiser=denoiser,
        lvae_model=lvae_model,
        bae_model=bae_model,
        sampler=sampler,
        w_dim=cfg.w_dim, c_dim=cfg.c_dim,
        w_pressure=cfg.w_pressure, w_aoa=cfg.w_aoa,
        lvae_params_dim=cfg.c_dim,
        params_mean_std=params_mean_std, aoas_mean_std=aoas_mean_std,
        name=cfg.model_name, opt_lr=cfg.lr,
    )
    ddm_w.w_mean     = w_mean
    ddm_w.w_std      = w_std
    ddm_w.p_ddm_mean = p_mean
    ddm_w.p_ddm_std  = p_std

    ddm_w.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ddm_w.optimizer, mode='min', factor=0.5, patience=600,
    )

    print(f"Denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, entity="adelbeke-",
                   name=cfg.model_name, config=vars(cfg))

    # 6. Training loop
    best_val = float('inf')
    print(f"Training for {cfg.n_epochs} epochs...")

    for epoch in range(cfg.n_epochs):
        train_loss, lw, laoa, lp = train_one_epoch(ddm_w, loader, cfg.device, cfg.grad_clip)
        val_loss,  vlw, vlaoa, vlp = eval_one_epoch(ddm_w, val_loader, cfg.device)
        ddm_w.scheduler.step(train_loss if math.isfinite(train_loss) else best_val)

        print(f"Epoch {epoch+1:05d}/{cfg.n_epochs} | "
              f"Train {train_loss:.4f} (w {lw:.4f} aoa {laoa:.4f} p {lp:.4f}) | "
              f"Val {val_loss:.4f} (w {vlw:.4f} aoa {vlaoa:.4f} p {vlp:.4f})")

        if use_wandb:
            wandb.log({"epoch": epoch+1,
                       "train_loss": train_loss, "train_w": lw,
                       "train_aoa": laoa, "train_p": lp,
                       "val_loss": val_loss, "val_w": vlw,
                       "val_aoa": vlaoa, "val_p": vlp})

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            ddm_w.save(cfg.save_dir, suffix="_best")

        if (epoch + 1) % cfg.save_every == 0:
            ddm_w.save(cfg.save_dir)

    ddm_w.save(cfg.save_dir)
    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
