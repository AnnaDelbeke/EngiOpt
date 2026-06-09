#!/bin/bash
#SBATCH --job-name=train_lvae_3d_v8_wp2
#SBATCH --output=logs/train_lvae_3d_v8_wp2_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --model_name lvae_3d_v8_wp2 \
    --joint_encoder \
    --pressure_embed_dim 16 \
    --dropout 0.25 \
    --lambda_lv 0.0 \
    --w_bae 1000.0 \
    --w_eta 1000.0 \
    --w_aoa 9.0 \
    --w_pressure 2.0 \
    --wandb \
    --wandb_project engiopt-lvae-3d
