#!/bin/bash
#SBATCH --job-name=plot_bae_3d_50slices
#SBATCH --output=logs/plot_bae_3d_50slices_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

CKPT="results/bezier_ae_3d_50slices/run_002/models/bezier_ae_3d_best.pt"
RUN_DIR="results/bezier_ae_3d_50slices/run_002"
SLICES="Wing_TL/data/processed/new_dataset_50slices_slices.pkl"
SCALARS="Wing_TL/data/processed/new_dataset_50slices_scalars.pkl"

python -u -m engiopt.bezier_ae.plot_thesis_bae3d_5panel \
    --checkpoint  $CKPT \
    --run_dir     $RUN_DIR \
    --slices_pkl  $SLICES \
    --scalars_pkl $SCALARS \
    --case_nums 40 \
    --wing_indices 5 10 20 30

python -u -m engiopt.bezier_ae.plot_thesis_bae3d_3d_views \
    --checkpoint  $CKPT \
    --run_dir     $RUN_DIR \
    --slices_pkl  $SLICES \
    --scalars_pkl $SCALARS \
    --case_nums 40 \
    --wing_indices 1 2 3 4
