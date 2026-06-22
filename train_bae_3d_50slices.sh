#!/bin/bash
#SBATCH --job-name=train_bae_3d_50slices
#SBATCH --output=logs/train_bae_3d_50slices_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.bezier_ae.train_bezier_ae_3d \
    --slices_pkl  Wing_TL/data/processed/new_dataset_50slices_slices.pkl \
    --scalars_pkl Wing_TL/data/processed/new_dataset_50slices_scalars.pkl \
    --results_dir results/bezier_ae_3d_50slices \
    --n_control_points 32 \
    --latent_dim 128 \
    --n_epochs 2000
