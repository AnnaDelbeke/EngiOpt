#!/bin/bash
#SBATCH --job-name=train_lvae_3d_v14
#SBATCH --output=logs/train_lvae_3d_v14_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --model_name lvae_3d_v14 \
    --joint_encoder \
    --lae_latent_dim 128 \
    --pressure_embed_dim 16 \
    --dropout 0.25 \
    --lambda_lv 1e-2 \
    --w_bae 100.0 \
    --w_eta 100.0 \
    --w_aoa 9.0 \
    --w_pressure 1.0 \
    --prune_every 500 \
    --prune_threshold 0.02 \
    --wandb \
    --wandb_project engiopt-lvae-3d
