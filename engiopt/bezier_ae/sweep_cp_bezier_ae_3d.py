"""
Sweep n_control_points in {8, 16, 32, 48, 64} for BezierAutoencoder3D.

Each value gets its own run_XXX directory under results/bezier_ae_3d/.
A summary CSV is written to results/bezier_ae_3d/cp_sweep_summary.csv.

Usage:
    python -m engiopt.bezier_ae.sweep_cp_bezier_ae_3d
"""

import csv
import os

import torch
from torch.utils.data import random_split

from engiopt.bezier_ae.train_bezier_ae_3d import (
    WingsBezierDataset3D,
    make_next_run_dir,
    train_one_config,
)
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

CP_VALUES = [8, 16, 32, 48, 64]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build dataset once; all sweep runs share the same split
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    train_items  = [item for item in list(new_dataset["train"]) if item["final"] == 1]
    val_items    = [item for item in list(new_dataset["val"])   if item["final"] == 1]
    train_ds     = WingsBezierDataset3D(train_items)
    val_ds       = WingsBezierDataset3D(val_items)
    print(f"Train wings: {len(train_ds)}  Val wings: {len(val_ds)}")

    n_spans = train_ds[0].shape[0]

    summary = []

    for n_cp in CP_VALUES:
        print(f"\n{'='*60}")
        print(f"  Sweep: n_control_points = {n_cp}")
        print(f"{'='*60}")

        run_dir = make_next_run_dir()
        # Write a small metadata file so the run is self-documenting
        with open(os.path.join(run_dir, "sweep_config.txt"), "w") as f:
            f.write(f"n_control_points: {n_cp}\n")
            f.write(f"sweep: cp_sweep\n")

        best_val, best_epoch, run_dir = train_one_config(
            train_ds, val_ds, n_spans,
            n_control_points=n_cp,
            device=device,
            run_dir=run_dir,
        )
        summary.append({"n_cp": n_cp, "best_val_loss": best_val,
                         "best_epoch": best_epoch, "run_dir": run_dir})
        print(f"  → best val {best_val:.6f} at epoch {best_epoch}  ({run_dir})")

    # Print and save summary
    print(f"\n{'='*60}")
    print("  CP Sweep Summary")
    print(f"{'='*60}")
    print(f"  {'n_cp':>6}  {'best_val_loss':>14}  {'best_epoch':>11}  run_dir")
    for row in summary:
        print(f"  {row['n_cp']:>6}  {row['best_val_loss']:>14.6f}  {row['best_epoch']:>11}  {row['run_dir']}")

    summary_path = "results/bezier_ae_3d/cp_sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_cp", "best_val_loss", "best_epoch", "run_dir"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
