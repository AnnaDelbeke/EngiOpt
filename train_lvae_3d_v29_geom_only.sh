#!/bin/bash
#SBATCH --job-name=train_lvae_3d_v29_geom_only
#SBATCH --output=logs/train_lvae_3d_v29_geom_only_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

# v29 settings but with --no_pressure:
#   - geometry-only encoder (no joint pressure embedding)
#   - no pressure head in decoder
#   - all other hyperparameters identical to v29

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --model_name lvae_3d_v29_geom_only \
    --no_pressure \
    --lae_latent_dim 64 \
    --dropout 0.25 \
    --lambda_lv 5e-1 \
    --w_bae 1000.0 \
    --w_eta 1000.0 \
    --w_aoa 9.0 \
    --w_pressure 0.0 \
    --w_perf 1.0 \
    --prune_every 500 \
    --prune_threshold 0.02 \
    --n_epochs 10000 \
    --lr 1e-3 \
    --seed 0 \
    --save_dir results/lvae_3d_geom_only \
    --wandb \
    --wandb_project engiopt-lvae-3d
