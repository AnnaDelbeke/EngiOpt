#!/bin/bash
#SBATCH --job-name=plot_case40_3d
#SBATCH --output=logs/plot_dataset2_case40_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_thesis_dataset2_case40_3d.py
