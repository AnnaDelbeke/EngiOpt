#!/bin/bash
#SBATCH --job-name=plot_ablation_ddm_w
#SBATCH --output=logs/plot_ablation_ddm_w_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python plot_ablation.py --out_dir results
