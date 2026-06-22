"""BAE-3D data ablation: train on n_samples wings, save to results/bezier_ae_3d_ablation/nN_sS/"""

import argparse
import os
import csv
import torch
from torch.utils.data import DataLoader, random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D, loss_reg_fn_3d
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--n_epochs",  type=int, default=2000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | n_samples={args.n_samples} seed={args.seed}")

    out_dir = f"results/bezier_ae_3d_ablation/n{args.n_samples}_s{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/models", exist_ok=True)

    ds = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_items = [it for it in list(ds["train"]) if it["final"] == 1]

    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(all_items), generator=rng)[:args.n_samples].tolist()
    subset = [all_items[i] for i in idx]

    full_ds = WingsBezierDataset3D(subset)
    train_size = int(0.9 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    n_spans = full_ds[0].shape[0]
    model = BezierAutoencoder3D(
        n_spans=n_spans, n_control_points=32, n_data_points=192,
        slice_hidden_dims=[256, 256, 256], span_hidden_dims=[256, 256],
        latent_dim=128,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=1e-6)

    best_val, best_epoch = float("inf"), 0
    metrics = []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        tr = 0.0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            coords_pred, z = model(x)
            loss = loss_reg_fn_3d(coords_pred, x, reg_weight=0.01, curvature_weight=0.0003)
            loss += 1e-4 * torch.mean(z ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr += loss.item() * x.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                coords_pred, z = model(x)
                loss = loss_reg_fn_3d(coords_pred, x, reg_weight=0.01, curvature_weight=0.0003)
                loss += 1e-4 * torch.mean(z ** 2)
                vl += loss.item() * x.size(0)
        vl /= len(val_loader.dataset)
        scheduler.step(vl)

        metrics.append((epoch, tr, vl))
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Train {tr:.6f} | Val {vl:.6f}")

        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_spans": n_spans, "latent_dim": 128,
                "slice_hidden_dims": [256, 256, 256],
                "span_hidden_dims":  [256, 256],
                "epoch": epoch, "val_loss": vl,
            }, f"{out_dir}/models/bae_3d_ablation_n{args.n_samples}_s{args.seed}_best.pt")

    with open(f"{out_dir}/metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        w.writerows(metrics)

    print(f"Best val {best_val:.6f} at epoch {best_epoch}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
