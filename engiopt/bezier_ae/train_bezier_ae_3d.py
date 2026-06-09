"""
Training script for BezierAutoencoder3D.

Differences from train_bezier_ae.py:
  - Dataset yields full wings [S, 2, 192] instead of individual slices [2, 192]
  - Loss uses loss_reg_fn_3d (reconstruction + spanwise smoothness)
  - Saves to results/bezier_ae_3d/ so it never overwrites the 2D BAE checkpoint
"""

import os
import csv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D, loss_reg_fn_3d
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


class WingsBezierDataset3D(Dataset):
    """Each sample is a full wing [S, 2, 192], with extra interpolated slices at the tip."""

    def __init__(self, base_dataset, num_extra_tip_slices=0):
        self.samples = []

        for item in base_dataset:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]

            coords[:, :, 1] -= te_shifts.unsqueeze(1)               # remove TE y-shift
            te_x = coords[:, 0, 0]
            coords[:, :, 0] += (1.0 - te_x).unsqueeze(1)            # shift TE x to 1

            le_x = coords[:, :, 0].min(dim=1).values                # [S] leading-edge x per slice
            chord = 1.0 - le_x                                       # [S]
            coords[:, :, 0] = (coords[:, :, 0] - le_x.unsqueeze(1)) / chord.unsqueeze(1)  # x in [0,1]

            x = coords.permute(0, 2, 1)                              # [S, 2, 192]

            if num_extra_tip_slices == 0:
                self.samples.append(x)
                continue

            # --- INTERPOLATION BETWEEN SLICE 14 AND 15 ---
            main_wing = x[:-1]     # Slices 1 to 14 -> Shape: [14, 2, 192]
            slice_14  = x[-2]      # Shape: [2, 192]
            slice_15  = x[-1]      # Shape: [2, 192]

            alphas = torch.linspace(0, 1, steps=num_extra_tip_slices + 2, device=x.device)[1:-1]

            extra_slices = []
            for alpha in alphas:
                interp_slice = torch.lerp(slice_14, slice_15, alpha)
                extra_slices.append(interp_slice.unsqueeze(0))

            extra_slices = torch.cat(extra_slices, dim=0) # [num_extra_tip_slices, 2, 192]

            # Reassemble into a denser 3D wing profile array
            x_dense = torch.cat([main_wing, extra_slices, slice_15.unsqueeze(0)], dim=0)
            self.samples.append(x_dense)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def make_next_run_dir(base_dir="results/bezier_ae_3d"):
    os.makedirs(base_dir, exist_ok=True)
    existing = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            try:
                existing.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    next_idx = 1 if not existing else max(existing) + 1
    run_dir = os.path.join(base_dir, f"run_{next_idx:03d}")
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "training_curves"), exist_ok=True)
    return run_dir


def train_one_epoch(model, loader, optimizer, device, reg_weight):
    model.train()
    total = 0.0
    for x in loader:
        x = x.to(device)                               # [B, S, 2, 192]
        optimizer.zero_grad()
        
        # Capture the latent vector 'z' instead of discarding it with '_'
        coords_pred, z = model(x)
        
        # Base structural reconstruction loss
        loss_recon = loss_reg_fn_3d(coords_pred, x, reg_weight=reg_weight, curvature_weight=0.0003)
        
        # Latent Space L2 penalty to penalize wild node variances (Stops Overfitting)
        loss_latent = 1e-4 * torch.mean(z ** 2)
        loss = loss_recon + loss_latent
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, device, reg_weight):
    model.eval()
    total = 0.0
    for x in loader:
        x = x.to(device)
        
        # Capture the latent vector 'z'
        coords_pred, z = model(x)
        
        # Compute identical loss components
        loss_recon = loss_reg_fn_3d(coords_pred, x, reg_weight=reg_weight, curvature_weight=0.0003)
        loss_latent = 1e-4 * torch.mean(z ** 2)
        loss = loss_recon + loss_latent
        
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def train_one_config(
    train_ds,
    val_ds,
    n_spans: int,
    n_control_points: int = 32,
    latent_dim: int = 64,
    slice_hidden_dims: list[int] = None,
    span_hidden_dims: list[int] = None,
    batch_size: int = 16,
    reg_weight: float = 0.001,
    learning_rate: float = 1e-3,
    n_epochs: int = 2000,
    device: torch.device = None,
    run_dir: str = None,
):
    if slice_hidden_dims is None:
        slice_hidden_dims = [64, 32]
    if span_hidden_dims is None:
        span_hidden_dims = [64, 32]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if run_dir is None:
        run_dir = make_next_run_dir()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = BezierAutoencoder3D(
        n_spans=n_spans,
        n_control_points=n_control_points,
        n_data_points=192,
        slice_hidden_dims=slice_hidden_dims,
        span_hidden_dims=span_hidden_dims,
        latent_dim=latent_dim,
        cpx_bound=[0.0, 1.0],
        cpy_bound=[-0.75, 0.75],
    ).to(device)
    print(f"  n_control_points={n_control_points}  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_epoch    = -1
    train_losses, val_losses = [], []
    best_model_path = os.path.join(run_dir, "models", "bezier_ae_3d_best.pt")

    for epoch in range(n_epochs):
        tr_loss  = train_one_epoch(model, train_loader, optimizer, device, reg_weight)
        val_loss = evaluate_loss(model, val_loader, device, reg_weight)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:04d} | Train {tr_loss:.6f} | Val {val_loss:.6f} | LR {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            torch.save({
                "epoch":             epoch + 1,
                "model_state_dict":  model.state_dict(),
                "val_loss":          val_loss,
                "n_spans":           n_spans,
                "n_control_points":  model.n_control_points,
                "latent_dim":        model.latent_dim,
                "slice_hidden_dims": slice_hidden_dims,
                "span_hidden_dims":  span_hidden_dims,
                "cpx_bound":         model.cpx_bound,
                "cpy_bound":         model.cpy_bound,
            }, best_model_path)

    print(f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}")

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([i, tr, va])

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"BezierAutoencoder3D  n_cp={n_control_points}")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "training_curves", "loss_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    return best_val_loss, best_epoch, run_dir


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_dir = make_next_run_dir()
    print(f"Outputs → {run_dir}")

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items    = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
                    if item["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items)
    print(f"Total wings: {len(full_dataset)}")

    train_size = int(0.9 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    n_spans = full_dataset[0].shape[0]

    train_one_config(
        train_ds, val_ds, n_spans,
        n_control_points=32,
        device=device,
        run_dir=run_dir,
    )

    print(f"Done. All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
