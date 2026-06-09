"""
DDM-3D: Denoising Diffusion Model operating directly in the 3D BAE latent space.

Same architecture as DDM_W but without the LVAE — denoises z_bae [B, 64] directly.

Pipeline
--------
  Training:
    wing coords → frozen 3D BAE encoder → z_bae [B, 64]
    MLPDenoiser learns to denoise z_bae conditioned on flow params and z_init
    (encoding of the initial/unoptimised wing)

  Inference:
    sample z_noise ~ N(0,I) → denoise → z_bae_gen → frozen 3D BAE decoder → wing coords

Usage
-----
    python -m engiopt.ddm.train_ddm_3d \\
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \\
        --model_name ddm_3d_v1 \\
        --wandb
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.ddm.ddm_w.ddm_w import MLPDenoiser
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE 3D
    bae_checkpoint: str = ""
    bae_latent_dim: int = 64
    n_spans:        int = 15

    # Denoiser
    z_dim:       int   = 64       # = bae_latent_dim
    c_dim:       int   = 4
    hidden_dims: tuple = (512, 512, 512, 512)
    t_embed_dim: int   = 128
    dropout:     float = 0.0

    # Diffusion
    num_diffusion_steps: int  = 1000
    cosine_schedule:     bool = False

    # Training
    batch_size: int  = 64
    lr:         float = 1e-4
    n_epochs:   int   = 20000
    grad_clip:  float = 1.0
    save_every: int   = 200
    w_aoa:      float = 9.0

    # Outputs
    save_dir:   str = "results/ddm_3d"
    model_name: str = "ddm_3d_v1"
    seed:       int = 0
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ZDataset(Dataset):
    def __init__(self, z_opts, aoas, params, z_inits,
                 scaler_params=None, scaler_aoas=None):
        self.z_opts    = z_opts
        self.aoas      = aoas
        self.params    = params
        self.z_inits   = z_inits
        self.scaler_params = scaler_params
        self.scaler_aoas   = scaler_aoas

    def __len__(self): return len(self.z_opts)

    def __getitem__(self, idx):
        z_opt  = self.z_opts[idx].clone()
        aoa    = self.aoas[idx].clone()
        params = self.params[idx].clone()
        z_init = self.z_inits[idx].clone()
        if self.scaler_params is not None: params = self.scaler_params.transform(params)
        if self.scaler_aoas   is not None: aoa    = self.scaler_aoas.transform(aoa)
        return z_opt, aoa, params, z_init


# ---------------------------------------------------------------------------
# Pre-computation: coords → 3D BAE → z_bae
# ---------------------------------------------------------------------------

def precompute_z(dataset_items, initial_by_case, bae_model, device):
    """Encode each wing directly with the 3D BAE, no LVAE."""
    print(f"Pre-computing BAE-3D encodings for {len(dataset_items)} samples...")

    z_opts_list, z_inits_list, aoas_list, params_list = [], [], [], []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(dataset_items):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(dataset_items)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]

            # Centre & normalise coords (same as LVAE 3D pipeline)
            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            le_x  = coords_c[:, :, 0].min(dim=1).values
            chord = 1.0 - le_x
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            # Encode full wing: [1, S, 2, 192] → [1, bae_latent_dim]
            x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)
            z_bae  = bae_model.encode(x_wing).squeeze(0).cpu()              # [bae_latent_dim]
            z_opts_list.append(z_bae)

            # z_init: encode the initial (unoptimised) wing
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
                z_init  = bae_model.encode(x_init).squeeze(0).cpu()
            else:
                z_init = z_bae

            z_inits_list.append(z_init)

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            aoas_list.append(aoa.unsqueeze(0) if aoa.ndim == 0 else aoa)

            params_list.append(torch.tensor(
                [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            ))

    print("Pre-computation complete.")
    return (
        torch.stack(z_opts_list),   # [N, bae_latent_dim]
        torch.stack(aoas_list),     # [N, 1]
        torch.stack(params_list),   # [N, 4]
        torch.stack(z_inits_list),  # [N, bae_latent_dim]
    )


# ---------------------------------------------------------------------------
# BAE loader
# ---------------------------------------------------------------------------

def load_bae_3d(checkpoint: str, device: str, latent_dim: int = 64,
                n_spans: int = 15) -> BezierAutoencoder3D:
    model = BezierAutoencoder3D(
        n_spans=n_spans,
        latent_dim=latent_dim,
        slice_hidden_dims=[64, 32],
        span_hidden_dims=[64, 32],
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded 3D BAE (n_spans={n_spans}, latent_dim={latent_dim}) from {checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Simple DDM-3D wrapper (no LVAE, denoises z_bae directly)
# ---------------------------------------------------------------------------

class DDM3D:
    """Thin wrapper: MLPDenoiser + diffusion sampler, operating on z_bae."""

    def __init__(self, denoiser: MLPDenoiser, bae_model: BezierAutoencoder3D,
                 sampler, z_dim: int = 64, c_dim: int = 4,
                 w_aoa: float = 9.0,
                 params_mean_std=None, aoas_mean_std=None,
                 name: str = "ddm_3d_v1", opt_lr: float = 1e-4):
        self.denoiser  = denoiser
        self.bae_model = bae_model
        self.sampler   = sampler
        self.z_dim     = z_dim
        self.w_aoa     = w_aoa
        self.name      = name

        self.z_mean = None
        self.z_std  = None

        self.optimizer = torch.optim.Adam(denoiser.parameters(), lr=opt_lr)
        self.scheduler = None

        self.scaler_params = scaler(params_mean_std) if params_mean_std is not None else None
        self.scaler_aoas   = scaler(aoas_mean_std)   if aoas_mean_std   is not None else None
        self.params_mean_std = params_mean_std
        self.aoas_mean_std   = aoas_mean_std

    def _sinusoidal_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (math.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def _forward_diffusion(self, x, t):
        return self.sampler.schedule_x.forward_diffusion_sample(x, t, x.device)

    def loss(self, batch, return_components=False):
        z_opt, aoa, params, z_init = batch
        device = z_opt.device
        B = z_opt.shape[0]

        t = torch.randint(0, self.sampler.T, (B,), device=device)

        z_noisy,   z_noise   = self._forward_diffusion(z_opt, t)
        aoa_noisy, aoa_noise = self._forward_diffusion(aoa,   t)

        z_noise_pred, aoa_noise_pred = self.denoiser(
            z_noisy, aoa_noisy, params, z_init, t
        )

        loss_z   = nn.functional.mse_loss(z_noise_pred,   z_noise)
        loss_aoa = nn.functional.mse_loss(aoa_noise_pred, aoa_noise)
        total    = loss_z + self.w_aoa * loss_aoa

        if return_components:
            return total, loss_z, loss_aoa
        return total

    def save(self, save_dir: str, suffix: str = ""):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.name}{suffix}.pth")
        torch.save({
            "denoiser":        self.denoiser.state_dict(),
            "optimizer":       self.optimizer.state_dict(),
            "params_mean_std": self.params_mean_std,
            "aoas_mean_std":   self.aoas_mean_std,
            "z_mean":          self.z_mean,
            "z_std":           self.z_std,
            "z_dim":           self.z_dim,
        }, path)
        return path

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.denoiser.load_state_dict(ckpt["denoiser"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.z_mean = ckpt.get("z_mean")
        self.z_std  = ckpt.get("z_std")
        if ckpt.get("params_mean_std") is not None:
            self.scaler_params   = scaler(ckpt["params_mean_std"])
            self.params_mean_std = ckpt["params_mean_std"]
        if ckpt.get("aoas_mean_std") is not None:
            self.scaler_aoas   = scaler(ckpt["aoas_mean_std"])
            self.aoas_mean_std = ckpt["aoas_mean_std"]
        print(f"Loaded DDM3D from {path}")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model: DDM3D, loader: DataLoader, device: str, grad_clip: float):
    model.denoiser.train()
    total_loss = total_z = total_aoa = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        model.optimizer.zero_grad()
        loss, lz, laoa = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), grad_clip)
        model.optimizer.step()
        total_loss += loss.item()
        total_z    += lz.item()
        total_aoa  += laoa.item()
        n += 1
    d = max(n, 1)
    return total_loss / d, total_z / d, total_aoa / d


@torch.no_grad()
def eval_one_epoch(model: DDM3D, loader: DataLoader, device: str):
    model.denoiser.eval()
    total_loss = total_z = total_aoa = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        loss, lz, laoa = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item()
        total_z    += lz.item()
        total_aoa  += laoa.item()
        n += 1
    model.denoiser.train()
    d = max(n, 1)
    return total_loss / d, total_z / d, total_aoa / d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bae_checkpoint", type=str, required=True)
    p.add_argument("--model_name",  type=str, default=None)
    p.add_argument("--n_epochs",    type=int, default=None)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--w_aoa",       type=float, default=9.0)
    p.add_argument("--bae_latent_dim", type=int, default=64)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--wandb_project", type=str, default="engiopt-ddm-3d")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.bae_checkpoint = args.bae_checkpoint
    cfg.bae_latent_dim = args.bae_latent_dim
    cfg.z_dim          = args.bae_latent_dim
    cfg.lr             = args.lr
    cfg.grad_clip      = args.grad_clip
    cfg.w_aoa          = args.w_aoa
    cfg.dropout        = args.dropout
    cfg.seed           = args.seed
    if args.model_name  is not None: cfg.model_name = args.model_name
    if args.n_epochs    is not None: cfg.n_epochs   = args.n_epochs
    if args.hidden_dims is not None: cfg.hidden_dims = tuple(args.hidden_dims)

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Device: {cfg.device}  Model: {cfg.model_name}")

    # 1. Frozen 3D BAE
    bae_model = load_bae_3d(cfg.bae_checkpoint, cfg.device,
                             latent_dim=cfg.bae_latent_dim, n_spans=cfg.n_spans)

    # 2. Dataset
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train       = list(new_dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset    = [item for item in all_train if item["final"] == 1]
    all_val             = list(new_dataset["val"])
    val_initial_by_case = {item["case_num"]: item for item in all_val if item["initial"] == 1}
    val_dataset         = [item for item in all_val if item["final"] == 1]
    print(f"Train: {len(base_dataset)}  Val: {len(val_dataset)}")

    # 3. Condition normalisation
    all_params_np = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas_np     = np.array([float(item["alpha"]) for item in base_dataset])
    params_mean_std = (all_params_np.mean(0), all_params_np.std(0))
    aoas_mean_std   = (float(all_aoas_np.mean()), float(all_aoas_np.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    # 4. Pre-compute BAE encodings
    z_opts, aoas, params_all, z_inits = precompute_z(
        base_dataset, initial_by_case, bae_model, cfg.device
    )
    z_mean    = z_opts.mean(dim=0, keepdim=True)
    z_std     = z_opts.std(dim=0,  keepdim=True).clamp(min=1e-8)
    z_opts_n  = (z_opts  - z_mean) / z_std
    z_inits_n = (z_inits - z_mean) / z_std

    dataset = ZDataset(z_opts_n, aoas, params_all, z_inits_n,
                       scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers)

    # Val
    val_z, val_aoas, val_params, val_z_inits = precompute_z(
        val_dataset, val_initial_by_case, bae_model, cfg.device
    )
    val_z_n       = (val_z       - z_mean) / z_std
    val_z_inits_n = (val_z_inits - z_mean) / z_std
    val_ds = ZDataset(val_z_n, val_aoas, val_params, val_z_inits_n,
                      scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    # 5. Build denoiser + sampler
    denoiser = MLPDenoiser(
        w_dim=cfg.z_dim, c_dim=cfg.c_dim,
        hidden_dims=cfg.hidden_dims, t_embed_dim=cfg.t_embed_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)

    sampler = samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4, end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
    )

    model = DDM3D(
        denoiser=denoiser, bae_model=bae_model, sampler=sampler,
        z_dim=cfg.z_dim, c_dim=cfg.c_dim, w_aoa=cfg.w_aoa,
        params_mean_std=params_mean_std, aoas_mean_std=aoas_mean_std,
        name=cfg.model_name, opt_lr=cfg.lr,
    )
    model.z_mean = z_mean
    model.z_std  = z_std

    model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.5, patience=600,
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
        train_loss, lz, laoa = train_one_epoch(model, loader, cfg.device, cfg.grad_clip)
        val_loss, vlz, vlaoa = eval_one_epoch(model, val_loader, cfg.device)
        model.scheduler.step(train_loss if math.isfinite(train_loss) else best_val)

        lr = model.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:05d}/{cfg.n_epochs} | "
              f"Train {train_loss:.4f} (z {lz:.4f} aoa {laoa:.4f}) | "
              f"Val {val_loss:.4f} (z {vlz:.4f} aoa {vlaoa:.4f}) | LR {lr:.2e}")

        if use_wandb:
            wandb.log({"epoch": epoch + 1,
                       "train_loss": train_loss, "train_z": lz, "train_aoa": laoa,
                       "val_loss": val_loss, "val_z": vlz, "val_aoa": vlaoa})

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            model.save(cfg.save_dir, suffix="_best")

        if (epoch + 1) % cfg.save_every == 0:
            model.save(cfg.save_dir)

    model.save(cfg.save_dir)
    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
