#!/bin/bash
#SBATCH --job-name=train_bae_3d
#SBATCH --output=logs/train_bae_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.train_bezier_ae_3d
