#!/bin/bash
#SBATCH --job-name=plot_ablation_wing3d
#SBATCH --output=logs/plot_ablation_wing3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_ablation_wing3d_surface.py --out results/ablation_wing3d_surface.pdf --wing_idx 5
