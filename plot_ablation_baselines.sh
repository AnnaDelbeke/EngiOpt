#!/bin/bash
#SBATCH --job-name=plot_ablation_baselines
#SBATCH --output=logs/plot_ablation_baselines_%j.out
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G
#SBATCH --dependency=afterok:4394987

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_ablation_baselines.py
