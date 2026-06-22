#!/bin/bash
#SBATCH --job-name=plot_ddm_ablation_4panel
#SBATCH --output=logs/plot_ddm_ablation_4panel_%j.out
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=2G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python plot_ddm_ablation_4panel.py
