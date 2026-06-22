#!/bin/bash
#SBATCH --job-name=eval_bae3d_full
#SBATCH --output=logs/eval_bae3d_full_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -m engiopt.bezier_ae.evaluate_bezier_ae_3d_ablation \
    --checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --n_samples 767 \
    --seed 0 \
    --out_dir results/bezier_ae_3d_ablation/n767_s0
