#!/bin/bash
#SBATCH --job-name=eval_bae_3d_run041
#SBATCH --output=logs/eval_bae_3d_run041_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
    --checkpoint results/bezier_ae_3d/run_041/models/bezier_ae_3d_best.pt \
    --run_dir    results/bezier_ae_3d/run_041
