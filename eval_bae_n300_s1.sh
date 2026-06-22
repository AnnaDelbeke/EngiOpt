#!/bin/bash
#SBATCH --job-name=eval_bae_n300_s1
#SBATCH --output=logs/eval_bae_n300_s1_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -m engiopt.bezier_ae.evaluate_bezier_ae_3d_ablation \
    --checkpoint results/bezier_ae_3d_ablation/n300_s1/models/bae_3d_ablation_n300_s1_best.pt \
    --n_samples 300 \
    --seed 1 \
    --out_dir results/bezier_ae_3d_ablation/n300_s1
