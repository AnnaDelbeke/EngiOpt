#!/bin/bash
#SBATCH --job-name=plot_bae3d_ablation
#SBATCH --output=logs/plot_bae3d_ablation_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python plot_ablation_bae_3d.py
