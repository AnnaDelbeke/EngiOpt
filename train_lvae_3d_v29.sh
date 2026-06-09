#!/bin/bash
#SBATCH --job-name=train_lvae_3d_v29
#SBATCH --output=logs/train_lvae_3d_v29_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --model_name lvae_3d_v29 \
    --joint_encoder \
    --pressure_embed_dim 16 \
    --dropout 0.25 \
    --lambda_lv 5e-1 \
    --decoder_frob_max 10.0 \
    --w_bae 1000.0 \
    --w_eta 1000.0 \
    --w_aoa 9.0 \
    --w_pressure 10.0 \
    --prune_every 500 \
    --prune_threshold 0.02 \
    --wandb \
    --wandb_project engiopt-lvae-3d
