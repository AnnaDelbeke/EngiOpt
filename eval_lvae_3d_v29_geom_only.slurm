#!/bin/bash
#SBATCH --job-name=eval_lvae_3d_geom_only
#SBATCH --output=logs/eval_lvae_3d_geom_only_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --nice=0

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.lvae.evaluate_lvae_3d \
    --checkpoint results/lvae_3d_geom_only/lvae_3d_v29_geom_only_best.pth \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --save_dir results/lvae_3d_evaluation
