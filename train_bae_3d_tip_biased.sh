#!/bin/bash
#SBATCH --job-name=bae_3d_tip_biased
#SBATCH --output=logs/bae_3d_tip_biased_%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.train_bezier_ae_3d_tip_biased
