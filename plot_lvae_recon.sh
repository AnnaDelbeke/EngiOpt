#!/bin/bash
#SBATCH --job-name=plot_lvae_recon
#SBATCH --output=logs/plot_lvae_recon_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.lvae.plot_thesis_lvae3d_recon

python -u -m engiopt.lvae.plot_thesis_lvae3d_recon \
    --full_slices \
    --slices_per_row 3 \
    --save_stem thesis/figures/lvae_recon_shape_cp_full
