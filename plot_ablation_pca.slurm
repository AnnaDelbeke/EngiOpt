#!/bin/bash
#SBATCH --job-name=plot_ablation_pca
#SBATCH --output=logs/plot_ablation_pca_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

set -e
cd /cluster/home/adelbeke/EngiOpt
source .venv/bin/activate

mkdir -p logs thesis/figures

python plot_ablation_pca.py 2>&1
