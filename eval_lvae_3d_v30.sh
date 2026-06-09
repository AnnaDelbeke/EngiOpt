#!/bin/bash
#SBATCH --job-name=eval_lvae_3d_v30
#SBATCH --output=logs/eval_lvae_3d_v30_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.evaluate_lvae_3d     --checkpoint results/lvae_3d/lvae_3d_v30_best.pth     --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt
