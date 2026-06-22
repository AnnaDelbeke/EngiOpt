#!/bin/bash
#SBATCH --job-name=train_lvae_run048
#SBATCH --output=logs/train_lvae_run048_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

# LVAE v29 retrain using corrected BAE (run_048, trained on train split only).
# Identical hyperparameters to v29 — only the BAE checkpoint changes.

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_048/models/bezier_ae_3d_best.pt \
    --model_name lvae_3d_v29_run048 \
    --joint_encoder \
    --pressure_embed_dim 16 \
    --lae_latent_dim 64 \
    --dropout 0.25 \
    --lambda_lv 5e-1 \
    --w_bae 1000.0 \
    --w_eta 1000.0 \
    --w_aoa 9.0 \
    --w_pressure 1.0 \
    --w_perf 1.0 \
    --prune_every 500 \
    --prune_threshold 0.02 \
    --n_epochs 10000 \
    --lr 1e-3 \
    --seed 0
