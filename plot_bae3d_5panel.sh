#!/bin/bash
#SBATCH --job-name=plot_bae3d_5panel
#SBATCH --output=logs/plot_bae3d_5panel_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.bezier_ae.plot_thesis_bae3d_5panel \
    --checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --run_dir    results/bezier_ae_3d/run_039 \
    --wing_indices 0 5 10 20
