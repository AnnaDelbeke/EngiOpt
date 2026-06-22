#!/bin/bash
#SBATCH --job-name=plot_case40
#SBATCH --output=logs/plot_case40_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8G

set -e
cd /cluster/home/adelbeke/EngiOpt
source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate

mkdir -p logs thesis/figures

python plot_thesis_dataset2_case40.py
