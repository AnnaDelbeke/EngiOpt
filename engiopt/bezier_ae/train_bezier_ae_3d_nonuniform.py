"""
Train BezierAutoencoder3D with non-uniform spanwise slice selection.

Inboard slices (indices 0-8) are sampled sparsely; outboard slices (9-14)
are kept fully, reflecting the observed pattern that geometric variation
accelerates sharply near the wingtip.

Selected indices (0-based, sorted root→tip):
  [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14]  →  11 spans

Compare against run_040 (15 uniform spans, n_cp=32) to assess whether
information-aware sampling improves tip reconstruction per span used.

Usage:
    python -m engiopt.bezier_ae.train_bezier_ae_3d_nonuniform
"""

from engiopt.bezier_ae.train_bezier_ae_3d import (
    WingsBezierDataset3D,
    make_next_run_dir,
    train_one_config,
)
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
from torch.utils.data import random_split
import torch
import os

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

# Sparse inboard (every other), dense outboard (all)
NONUNIFORM_INDICES = [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Sub-slice indices: {NONUNIFORM_INDICES}  ({len(NONUNIFORM_INDICES)} spans)")

    run_dir = make_next_run_dir()
    print(f"Outputs → {run_dir}")

    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"sub_slice_indices: {NONUNIFORM_INDICES}\n")
        f.write(f"n_spans: {len(NONUNIFORM_INDICES)}\n")
        f.write(f"n_control_points: 32\n")
        f.write(f"note: non-uniform spanwise sampling, compare vs run_040\n")

    ds = NewWingsDataset(
        _SLICES_PKL, _SCALARS_PKL, seed=0,
        sub_slice_indices=NONUNIFORM_INDICES,
    )
    all_items = [i for i in list(ds["train"]) + list(ds["val"]) if i["final"] == 1]
    full_dataset = WingsBezierDataset3D(all_items)
    print(f"Total wings: {len(full_dataset)}, n_spans per wing: {full_dataset[0].shape[0]}")

    train_size = int(0.9 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    n_spans = full_dataset[0].shape[0]
    best_val, best_epoch, run_dir = train_one_config(
        train_ds, val_ds, n_spans,
        n_control_points=32,
        device=device,
        run_dir=run_dir,
    )
    print(f"Best val loss {best_val:.6f} at epoch {best_epoch}")
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
