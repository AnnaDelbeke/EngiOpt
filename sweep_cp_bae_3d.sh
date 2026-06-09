#!/bin/bash
#SBATCH --job-name=sweep_cp_bae_3d
#SBATCH --output=logs/sweep_cp_bae_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.sweep_cp_bezier_ae_3d
