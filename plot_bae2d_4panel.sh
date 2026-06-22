#!/bin/bash
#SBATCH --job-name=plot_bae2d_4panel
#SBATCH --output=logs/plot_bae2d_4panel_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.plot_thesis_bae2d_5panel
