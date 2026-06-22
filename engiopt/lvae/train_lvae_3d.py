"""
Training script for the LVAE using BezierAutoencoder3D latents.

Differences from train_lvae.py:
  - BAE produces a single joint latent [B, bae_latent_dim] per wing (not [B, 9, 3, 30])
  - LAEEncoder3D / LAEDecoder3D replace the original UNet-style encoder/decoder
  - Saves to results/lvae_3d/ — never touches existing LVAE checkpoints

Usage
-----
    python -m engiopt.lvae.train_lvae_3d \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --model_name lvae_3d_v1 \
        --flow_only \
        --wandb
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


# ---------------------------------------------------------------------------
# Encoder / Decoder (MLP-based, takes joint BAE latent)
# ---------------------------------------------------------------------------

class LAEEncoder3D(nn.Module):
    """Encodes BAE latent + flow params → LVAE latent w.

    Input:  [B, bae_latent_dim + c_dim]
    Output: [B, lae_latent_dim]
    """

    def __init__(self, bae_latent_dim: int = 256, c_dim: int = 4,
                 n_spans: int = 15, pressure_length: int = 192,
                 lae_latent_dim: int = 64,
                 hidden_dims: list[int] = [512, 512, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.lae_latent_dim   = lae_latent_dim
        self.dropout_p        = dropout
        self.n_spans          = n_spans
        self.pressure_length  = pressure_length
        input_dim = bae_latent_dim + c_dim
        dims = [input_dim] + hidden_dims + [lae_latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z_bae: torch.Tensor, pressure: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:
        """
        z_bae    : [B, bae_latent_dim]
        pressure : [B, n_spans, pressure_length]  (unused — geometry-only encoder)
        params   : [B, c_dim]
        returns  : [B, lae_latent_dim]
        """
        x = torch.cat([z_bae, params], dim=1)
        return self.net(x)


class LAEEncoderJoint3D(nn.Module):
    """Encodes BAE latent + compressed pressure + flow params → LVAE latent w.

    Pressure is first compressed per-span with a shared MLP (192 → pressure_embed_dim),
    then flattened and concatenated with z_bae and params before the main encoder.
    This keeps the encoder truly joint without the 2880-dim overfitting problem.

    Input:  z_bae [B, bae_latent_dim]  +  pressure [B, n_spans, 192]  +  params [B, c_dim]
    Output: [B, lae_latent_dim]
    """

    def __init__(self, bae_latent_dim: int = 64, c_dim: int = 4,
                 n_spans: int = 15, pressure_length: int = 192,
                 pressure_embed_dim: int = 16,
                 lae_latent_dim: int = 64,
                 hidden_dims: list[int] = [512, 512, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.lae_latent_dim     = lae_latent_dim
        self.dropout_p          = dropout
        self.n_spans            = n_spans
        self.pressure_length    = pressure_length
        self.pressure_embed_dim = pressure_embed_dim

        # shared MLP applied independently to each span's pressure [192] → [pressure_embed_dim]
        self.pressure_encoder = nn.Sequential(
            nn.Linear(pressure_length, 64),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(64, pressure_embed_dim),
            nn.GELU(),
        )

        input_dim = bae_latent_dim + n_spans * pressure_embed_dim + c_dim
        dims = [input_dim] + hidden_dims + [lae_latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z_bae: torch.Tensor, pressure: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:
        """
        z_bae    : [B, bae_latent_dim]
        pressure : [B, n_spans, pressure_length]
        params   : [B, c_dim]
        returns  : [B, lae_latent_dim]
        """
        B, S, L = pressure.shape
        # compress each span independently: [B*S, L] → [B*S, embed] → [B, S*embed]
        p_embed = self.pressure_encoder(pressure.reshape(B * S, L))
        p_flat  = p_embed.reshape(B, S * self.pressure_embed_dim)
        x = torch.cat([z_bae, p_flat, params], dim=1)
        return self.net(x)


class LAEDecoder3D(nn.Module):
    """Decodes LVAE latent w + flow params → BAE latent + pressure + AoA + eta_y.

    Mirrors what the original LAEDecoder produces so DDM_W can stay the same
    structure, but outputs bae_latent instead of [B, S, 3, 30].
    """

    def __init__(self, bae_latent_dim: int = 256, c_dim: int = 4,
                 lae_latent_dim: int = 64, n_spans: int = 9,
                 pressure_length: int = 192,
                 hidden_dims: list[int] = [256, 512, 512],
                 dropout: float = 0.0,
                 use_perf_head: bool = True,
                 span_embed_dim: int = 8,
                 spectral_norm: bool = False,
                 use_pressure: bool = True):
        super().__init__()
        self.bae_latent_dim  = bae_latent_dim
        self.n_spans         = n_spans
        self.pressure_length = pressure_length
        self.use_pressure    = use_pressure
        _sn = sn if spectral_norm else (lambda x: x)

        dims = [lae_latent_dim + c_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(_sn(nn.Linear(dims[i], dims[i + 1])))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        h = hidden_dims[-1]

        self.bae_head   = _sn(nn.Linear(h, bae_latent_dim))
        self.aoa_head   = _sn(nn.Linear(h, 1))
        self.eta_y_head = _sn(nn.Linear(h, n_spans))
        self.perf_head  = _sn(nn.Linear(h, 2)) if use_perf_head else None
        if use_pressure:
            self.pressure_head = nn.Sequential(
                _sn(nn.Linear(h, 512)),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                _sn(nn.Linear(512, n_spans * pressure_length)),
            )
        else:
            self.pressure_head = None

    def forward(self, w: torch.Tensor, params: torch.Tensor):
        """
        w      : [B, lae_latent_dim]
        params : [B, c_dim]
        returns: (z_bae_pred [B, bae_latent_dim],
                  aoa_pred   [B, 1],
                  eta_y_pred [B, n_spans, 1],
                  pressure_pred [B, n_spans, pressure_length],
                  perf_pred  [B, 2])
        """
        B = w.shape[0]
        x = torch.cat([w, params], dim=1)
        h = self.backbone(x)

        z_bae_pred    = self.bae_head(h)
        aoa_pred      = self.aoa_head(h)
        eta_y_pred    = self.eta_y_head(h).reshape(B, self.n_spans, 1)
        pressure_pred = (self.pressure_head(h).reshape(B, self.n_spans, self.pressure_length)
                         if self.use_pressure else None)
        perf_pred     = self.perf_head(h) if self.perf_head is not None else None

        return z_bae_pred, aoa_pred, eta_y_pred, pressure_pred, perf_pred


# ---------------------------------------------------------------------------
# LVAE3D wrapper (mirrors LAE_AoAInit interface used by DDM_W)
# ---------------------------------------------------------------------------

class LVAE3D(nn.Module):
    """Thin wrapper around encoder + decoder, mirrors the LAE_AoAInit API."""

    def __init__(self, encoder: LAEEncoder3D, decoder: LAEDecoder3D,
                 bae_model: BezierAutoencoder3D,
                 lae_latent_dim: int = 64,
                 params_mean_std=None, aoas_mean_std=None,
                 pressures_mean_std=None,
                 name: str = "lvae_3d_v1",
                 lambda_lv: float = 7e-5,
                 weights: tuple = (1.0, 9.0, 1.0, 1.0, 1.0),
                 opt_lr: float = 1e-3):
        super().__init__()
        self.encoder        = encoder
        self.decoder        = decoder
        self.bae_model      = bae_model
        self.lae_latent_dim = lae_latent_dim
        self.name           = name
        self.lambda_lv      = lambda_lv
        self.w_bae, self.w_aoa, self.w_eta, self.w_pressure, self.w_perf = weights

        from engiopt.data_processing.utils import scaler as make_scaler
        self.scaler_params    = make_scaler(params_mean_std)    if params_mean_std    else None
        self.scaler_aoas      = make_scaler(aoas_mean_std)      if aoas_mean_std      else None
        self.scaler_pressures = make_scaler(pressures_mean_std) if pressures_mean_std else None
        self.scaler_perfs     = None   # set after construction once perf stats are known

        # Active latent mask (all dims active initially)
        self.register_buffer(
            "active_latent_mask",
            torch.ones(lae_latent_dim, dtype=torch.bool),
        )

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=opt_lr
        )
        self.stats = {
            'train_loss': np.array([]),
            'train_loss_epoch': np.array([]),
            'test_loss_recon': [],
            'current_loss': 0.0,
            'best_loss': float('inf'),
        }

    def _apply_mask(self, w: torch.Tensor) -> torch.Tensor:
        mask = self.active_latent_mask.to(w.device)
        return w * mask.float()

    @torch.no_grad()
    def update_active_mask(self, loader, device, threshold: float = 0.02):
        """Zero out latent dims whose std across the dataset is below threshold."""
        self.encoder.eval()
        all_w = []
        for batch in loader:
            z_bae, _, params, _, pressure, _ = [t.to(device) for t in batch]
            all_w.append(self.encoder(z_bae, pressure, params).cpu())
        self.encoder.train()
        std = torch.cat(all_w, dim=0).std(dim=0)
        self.active_latent_mask = (std >= threshold).to(device)
        n_active = int(self.active_latent_mask.sum().item())
        print(f"  [MASK] {n_active}/{self.lae_latent_dim} active dims  "
              f"(threshold={threshold})")

    def encode(self, z_bae: torch.Tensor, pressure: torch.Tensor,
               params: torch.Tensor) -> torch.Tensor:
        return self.encoder(z_bae, pressure, params)

    def decode(self, w: torch.Tensor, params: torch.Tensor):
        return self.decoder(w, params)

    def _loss(self, batch, device):
        z_bae, aoa, params, eta_y, pressure, perf = [t.to(device) for t in batch]

        w = self.encoder(z_bae, pressure, params).clamp(-10.0, 10.0)
        w_masked = self._apply_mask(w)
        z_bae_pred, aoa_pred, eta_y_pred, pressure_pred, perf_pred = self.decoder(w_masked, params)

        loss_bae      = nn.functional.mse_loss(z_bae_pred, z_bae)
        loss_aoa      = nn.functional.mse_loss(aoa_pred, aoa)
        loss_eta      = nn.functional.mse_loss(eta_y_pred, eta_y)
        loss_pressure = (nn.functional.mse_loss(pressure_pred, pressure)
                         if pressure_pred is not None
                         else torch.zeros(1, device=device))
        loss_perf = (nn.functional.mse_loss(perf_pred, perf)
                     if perf_pred is not None
                     else torch.zeros(1, device=device))

        # Least-volume penalty: product of per-dim stds (log-sum-exp for stability)
        # Clamp w to prevent encoder from cheating via extreme activations
        w_clamped = w.clamp(-10.0, 10.0)
        loss_lv = torch.exp(torch.log(w_clamped.std(dim=0) + 1e-4).mean())

        total = (self.w_bae      * loss_bae
                 + self.w_aoa      * loss_aoa
                 + self.w_eta      * loss_eta
                 + self.w_pressure * loss_pressure
                 + self.w_perf     * loss_perf
                 + self.lambda_lv  * loss_lv)
        return total, loss_bae, loss_aoa, loss_eta, loss_pressure, loss_perf

    def _update(self, batch, device):
        self.encoder.train(); self.decoder.train()
        self.optimizer.zero_grad()
        loss, *_ = self._loss(batch, device)
        loss.backward()
        self.optimizer.step()
        self.stats['current_loss'] = loss.item()

    def save(self, save_dir: str, suffix: str = ""):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.name}{suffix}.pth")
        torch.save({
            'encoder':           self.encoder.state_dict(),
            'decoder':           self.decoder.state_dict(),
            'optimizer':         self.optimizer.state_dict(),
            'stats':             self.stats,
            'active_latent_mask': self.active_latent_mask,
            'n_spans':           self.decoder.n_spans,
            'pressure_length':      getattr(self.encoder, 'pressure_length', 192),
            'pressure_embed_dim':   getattr(self.encoder, 'pressure_embed_dim', None),
            'span_embed_dim':       None,
            'point_embed':          False,
            'joint_encoder':        isinstance(self.encoder, LAEEncoderJoint3D),
            'use_pressure':         getattr(self.decoder, 'use_pressure', True),
            'lae_latent_dim':       self.lae_latent_dim,
            'bae_latent_dim':       self.decoder.bae_latent_dim,
            'dropout':              getattr(self.encoder, 'dropout_p', 0.0),
            'params_mean_std':   (self.scaler_params.mean, self.scaler_params.std)
                                  if self.scaler_params else None,
            'aoas_mean_std':     (self.scaler_aoas.mean, self.scaler_aoas.std)
                                  if self.scaler_aoas else None,
            'perfs_mean_std':     (self.scaler_perfs.mean, self.scaler_perfs.std)
                                  if self.scaler_perfs else None,
            'pressures_mean_std': (self.scaler_pressures.mean, self.scaler_pressures.std)
                                   if self.scaler_pressures else None,
        }, path)
        return path

    def load(self, path: str, train_mode: bool = False):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        if train_mode and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'active_latent_mask' in ckpt:
            self.active_latent_mask = ckpt['active_latent_mask']
        self.stats = ckpt.get('stats', self.stats)
        if not train_mode:
            self.encoder.eval(); self.decoder.eval()
        print(f"Loaded LVAE3D from {path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrecomputedWingsDataset3D(Dataset):
    """Pre-encoded 3D BAE latents."""

    def __init__(self, z_baes, aoas, params, eta_ys, pressures, perfs,
                 scaler_params=None, scaler_aoas=None, scaler_pressures=None,
                 scaler_perfs=None):
        self.z_baes    = z_baes      # [N, bae_latent_dim]
        self.aoas      = aoas        # [N, 1]
        self.params    = params      # [N, c_dim]
        self.eta_ys    = eta_ys      # [N, n_spans, 1]
        self.pressures = pressures   # [N, n_spans, 192]
        self.perfs     = perfs       # [N, 2]
        self.scaler_params    = scaler_params
        self.scaler_aoas      = scaler_aoas
        self.scaler_pressures = scaler_pressures
        self.scaler_perfs     = scaler_perfs

    def __len__(self):
        return len(self.z_baes)

    def __getitem__(self, idx):
        z_bae    = self.z_baes[idx].clone()
        aoa      = self.aoas[idx].clone()
        params   = self.params[idx].clone()
        eta_y    = self.eta_ys[idx].clone()
        pressure = self.pressures[idx].clone()
        perf     = self.perfs[idx].clone()
        if self.scaler_params    is not None: params   = self.scaler_params.transform(params)
        if self.scaler_aoas      is not None: aoa      = self.scaler_aoas.transform(aoa)
        if self.scaler_pressures is not None: pressure = self.scaler_pressures.transform(pressure)
        if self.scaler_perfs     is not None: perf     = self.scaler_perfs.transform(perf)
        return z_bae, aoa, params, eta_y, pressure, perf


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute_latents_3d(base_dataset, bae_model: BezierAutoencoder3D, device):
    """Encode entire dataset with frozen 3D BAE → joint latents."""
    print(f"Pre-computing 3D BAE latents for {len(base_dataset)} samples...")
    z_baes_list, aoas_list, params_list, eta_ys_list, pressures_list, perfs_list = [], [], [], [], [], []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(base_dataset):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(base_dataset)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]
            eta_y     = te_shifts.unsqueeze(1)                                # [S, 1]

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            le_x  = coords_c[:, :, 0].min(dim=1).values        # [S]
            chord = 1.0 - le_x                                   # [S]
            coords_c[:, :, 0] = (coords_c[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)

            # [S, 192, 2] → [1, S, 2, 192]
            x_wing = coords_c.permute(0, 2, 1).unsqueeze(0).to(device)
            z_bae  = bae_model.encode(x_wing).squeeze(0).cpu()   # [bae_latent_dim]
            z_baes_list.append(z_bae)
            eta_ys_list.append(eta_y)

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            aoas_list.append(aoa.unsqueeze(0) if aoa.ndim == 0 else aoa)

            flow_params = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            params_list.append(torch.tensor(flow_params, dtype=torch.float32))

            pressure = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)
            pressures_list.append(pressure)

            perf = torch.tensor([float(item["cd_val"]), float(item["cl_val"])], dtype=torch.float32)
            perfs_list.append(perf)

    print("Pre-computation complete.")
    return (
        torch.stack(z_baes_list),     # [N, bae_latent_dim]
        torch.stack(aoas_list),        # [N, 1]
        torch.stack(params_list),      # [N, 4]
        torch.stack(eta_ys_list),      # [N, S, 1]
        torch.stack(pressures_list),   # [N, S, 192]
        torch.stack(perfs_list),       # [N, 2]
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_bae_3d(bae_checkpoint: str, device: str,
                latent_dim: int = 256) -> BezierAutoencoder3D:
    ckpt              = torch.load(bae_checkpoint, map_location=device, weights_only=False)
    n_spans           = ckpt["n_spans"]
    latent_dim        = ckpt.get("latent_dim", latent_dim)
    slice_hidden_dims = ckpt.get("slice_hidden_dims", [256, 128])
    span_hidden_dims  = ckpt.get("span_hidden_dims",  [256, 128])
    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=32, n_data_points=192,
        slice_hidden_dims=slice_hidden_dims, span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded 3D BAE from {bae_checkpoint}  (n_spans={n_spans}, latent_dim={latent_dim})")
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_loss(model: LVAE3D, loader: DataLoader, device: str):
    model.encoder.eval(); model.decoder.eval()
    total = 0.0
    for batch in loader:
        loss, *_ = model._loss(batch, device)
        total += loss.item()
    model.encoder.train(); model.decoder.train()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bae_checkpoint", type=str, required=True,
                   help="Path to trained 3D BAE checkpoint (bezier_ae_3d_best.pt)")
    p.add_argument("--model_name",  type=str, default="lvae_3d_v1")
    p.add_argument("--n_epochs",    type=int, default=10000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--bae_latent_dim", type=int, default=256)
    p.add_argument("--lae_latent_dim", type=int, default=64)
    p.add_argument("--lambda_lv",   type=float, default=0.0)
    p.add_argument("--w_bae",       type=float, default=1000.0)
    p.add_argument("--w_aoa",       type=float, default=9.0)
    p.add_argument("--w_eta",       type=float, default=1000.0)
    p.add_argument("--w_pressure",  type=float, default=1.0)
    p.add_argument("--w_perf",      type=float, default=1.0)
    p.add_argument("--joint_encoder",     action="store_true",
                   help="Use LAEEncoderJoint3D (pressure pre-encoder) instead of geometry-only")
    p.add_argument("--no_pressure",       action="store_true",
                   help="Remove pressure head from decoder and force geometry-only encoder "
                        "(overrides --joint_encoder). Trains geometry-only with all other v29 settings.")
    p.add_argument("--pressure_embed_dim", type=int, default=16,
                   help="Per-span pressure embedding dim (only used with --joint_encoder)")
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--prune_every",      type=int,   default=500)
    p.add_argument("--prune_threshold",  type=float, default=0.02)
    p.add_argument("--decoder_frob_max", type=float, default=0.0,
                   help="Hard Frobenius norm ceiling per decoder weight matrix (0=disabled)")
    p.add_argument("--n_samples",         type=int,   default=0,
                   help="Subsample training set to this many wings (0 = use all)")
    p.add_argument("--save_dir",         type=str,   default="results/lvae_3d")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--wandb_project", type=str, default="engiopt-lvae-3d")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Frozen 3D BAE — dims are read from checkpoint, args.bae_latent_dim is ignored
    bae_model = load_bae_3d(args.bae_checkpoint, device)
    args.bae_latent_dim = bae_model.latent_dim

    # 2. Dataset
    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_train       = list(new_dataset["train"])
    base_dataset    = [item for item in all_train if item["final"] == 1]
    all_val         = list(new_dataset["val"])
    val_dataset_raw = [item for item in all_val  if item["final"] == 1]
    if args.n_samples > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(base_dataset), size=min(args.n_samples, len(base_dataset)), replace=False)
        base_dataset = [base_dataset[i] for i in sorted(idx)]
    print(f"Train: {len(base_dataset)}  Val: {len(val_dataset_raw)}")

    # 3. Normalisation stats
    all_params_np = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas_np = np.array([float(item["alpha"]) for item in base_dataset])
    params_mean_std = (all_params_np.mean(0), all_params_np.std(0))
    aoas_mean_std   = (float(all_aoas_np.mean()), float(all_aoas_np.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    # 4. Pre-compute
    z_baes, aoas, params_all, eta_ys, pressures, perfs = precompute_latents_3d(
        base_dataset, bae_model, device
    )
    z_baes_val, aoas_val, params_val, eta_ys_val, pressures_val, perfs_val = precompute_latents_3d(
        val_dataset_raw, bae_model, device
    )

    pressure_mean    = float(pressures.mean())
    pressure_std     = float(pressures.std())
    scaler_pressures = scaler((pressure_mean, pressure_std))

    perfs_mean_std = (perfs.mean(0).numpy(), perfs.std(0).numpy())
    scaler_perfs   = scaler(perfs_mean_std)

    batch_size = 32
    dataset    = PrecomputedWingsDataset3D(
        z_baes, aoas, params_all, eta_ys, pressures, perfs,
        scaler_params=scaler_params, scaler_aoas=scaler_aoas,
        scaler_pressures=scaler_pressures, scaler_perfs=scaler_perfs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_ds = PrecomputedWingsDataset3D(
        z_baes_val, aoas_val, params_val, eta_ys_val, pressures_val, perfs_val,
        scaler_params=scaler_params, scaler_aoas=scaler_aoas,
        scaler_pressures=scaler_pressures, scaler_perfs=scaler_perfs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Build model
    n_spans = eta_ys.shape[1]
    pressure_length = pressures.shape[2]  # 192
    use_pressure = not args.no_pressure
    if args.no_pressure:
        # --no_pressure forces geometry-only encoder regardless of --joint_encoder
        encoder = LAEEncoder3D(
            bae_latent_dim=args.bae_latent_dim, c_dim=4,
            n_spans=n_spans, pressure_length=pressure_length,
            lae_latent_dim=args.lae_latent_dim, dropout=args.dropout,
        ).to(device)
        print("=== NO_PRESSURE: geometry-only encoder, no pressure head in decoder ===")
    elif args.joint_encoder:
        encoder = LAEEncoderJoint3D(
            bae_latent_dim=args.bae_latent_dim, c_dim=4,
            n_spans=n_spans, pressure_length=pressure_length,
            pressure_embed_dim=args.pressure_embed_dim,
            lae_latent_dim=args.lae_latent_dim, dropout=args.dropout,
        ).to(device)
        print(f"Using joint encoder (pressure_embed_dim={args.pressure_embed_dim})")
    else:
        encoder = LAEEncoder3D(
            bae_latent_dim=args.bae_latent_dim, c_dim=4,
            n_spans=n_spans, pressure_length=pressure_length,
            lae_latent_dim=args.lae_latent_dim, dropout=args.dropout,
        ).to(device)
        print("Using geometry-only encoder")
    decoder = LAEDecoder3D(
        bae_latent_dim=args.bae_latent_dim, c_dim=4,
        lae_latent_dim=args.lae_latent_dim, n_spans=n_spans,
        dropout=args.dropout,
        spectral_norm=(args.lambda_lv > 0),
        use_pressure=use_pressure,
    ).to(device)

    model = LVAE3D(
        encoder=encoder, decoder=decoder, bae_model=bae_model,
        lae_latent_dim=args.lae_latent_dim,
        params_mean_std=params_mean_std, aoas_mean_std=aoas_mean_std,
        pressures_mean_std=(pressure_mean, pressure_std),
        name=args.model_name, lambda_lv=args.lambda_lv, opt_lr=args.lr,
        weights=(args.w_bae, args.w_aoa, args.w_eta, args.w_pressure, args.w_perf),
    )
    model.scaler_perfs = scaler_perfs

    n_params = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
    print(f"LVAE3D parameters: {n_params:,}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, patience=200, factor=0.5, min_lr=1e-5
    )

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, entity="adelbeke-",
                   name=args.model_name,
                   config=vars(args))

    best_val = float('inf')
    val_every  = 200
    save_every = 200

    print(f"Training for {args.n_epochs} epochs...")
    for epoch in range(args.n_epochs):
        model.encoder.train(); model.decoder.train()
        total_loss = 0.0
        loss_bae_sum = loss_aoa_sum = loss_eta_sum = loss_pressure_sum = loss_perf_sum = 0.0
        for batch in loader:
            model.optimizer.zero_grad()
            loss, l_bae, l_aoa, l_eta, l_pressure, l_perf = model._loss(batch, device)
            loss.backward()
            model.optimizer.step()
            # Hard Frobenius-norm ceiling on decoder weights — prevents wide matrices
            # from bypassing spectral norm via the Frobenius loophole
            if args.decoder_frob_max > 0:
                with torch.no_grad():
                    for name, param in model.decoder.named_parameters():
                        if 'weight' in name:
                            norm = param.norm()
                            if norm > args.decoder_frob_max:
                                param.mul_(args.decoder_frob_max / norm)
            model.stats['current_loss'] = loss.item()
            total_loss        += loss.item()
            loss_bae_sum      += l_bae.item()
            loss_aoa_sum      += l_aoa.item()
            loss_eta_sum      += l_eta.item()
            loss_pressure_sum += l_pressure.item()
            loss_perf_sum     += l_perf.item()
        n_batches  = max(len(loader), 1)
        train_loss = total_loss / n_batches
        scheduler.step(train_loss)
        lr = model.optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:05d}/{args.n_epochs} | Train {train_loss:.6f} | LR {lr:.2e}")

        if use_wandb:
            wandb.log({
                "epoch":         epoch + 1,
                "train_loss":    train_loss,
                "loss_bae":      loss_bae_sum      / n_batches,
                "loss_aoa":      loss_aoa_sum      / n_batches,
                "loss_eta":      loss_eta_sum      / n_batches,
                "loss_pressure": loss_pressure_sum  / n_batches,
                "loss_perf":     loss_perf_sum      / n_batches,
                "lr":            lr,
            })

        if (epoch + 1) % val_every == 0:
            val_loss = compute_val_loss(model, val_loader, device)
            n_active = int(model.active_latent_mask.sum().item())

            # Compute per-dim stds on training set for PCA-ordering monitoring
            model.encoder.eval()
            all_w = []
            with torch.no_grad():
                for batch in loader:
                    z_bae, _, params, _, pressure, _ = [t.to(device) for t in batch]
                    w = model.encoder(z_bae, pressure, params).clamp(-10.0, 10.0)
                    all_w.append(w.cpu())
            all_w = torch.cat(all_w, dim=0)
            dim_stds = all_w.std(dim=0)                        # [lae_latent_dim]
            stds_sorted = dim_stds.sort(descending=True).values
            model.encoder.train()

            print(f"  [VAL] {val_loss:.6f}  (best {best_val:.6f})  active_dims={n_active}"
                  f"  std_max={stds_sorted[0]:.3f}  std_min={stds_sorted[-1]:.3f}"
                  f"  ratio={stds_sorted[0]/stds_sorted[-1].clamp(min=1e-6):.1f}x")

            if use_wandb:
                log_dict = {
                    "epoch":       epoch + 1,
                    "val_loss":    val_loss,
                    "active_dims": n_active,
                    "latent_std_max":   stds_sorted[0].item(),
                    "latent_std_min":   stds_sorted[-1].item(),
                    "latent_std_ratio": (stds_sorted[0] / stds_sorted[-1].clamp(min=1e-6)).item(),
                }
                # Log each sorted dim std so wandb shows the full slope curve
                for i, s in enumerate(stds_sorted):
                    log_dict[f"latent_std/dim_{i:02d}"] = s.item()
                wandb.log(log_dict)
            if val_loss < best_val:
                best_val = val_loss
                model.save(args.save_dir, suffix="_best")
                print("  --> New best saved.")

        if args.prune_every > 0 and (epoch + 1) % args.prune_every == 0:
            model.update_active_mask(loader, device, threshold=args.prune_threshold)

        if (epoch + 1) % save_every == 0:
            model.save(args.save_dir)

    model.save(args.save_dir)
    print(f"Done. Best val loss: {best_val:.6f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
