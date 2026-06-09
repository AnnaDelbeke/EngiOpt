#!/bin/bash
#SBATCH --job-name=train_ddm_3d_do20
#SBATCH --output=logs/train_ddm_3d_do20_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.train_ddm_3d \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --model_name ddm_3d_do20 \
    --bae_latent_dim 64 \
    --w_aoa 9.0 \
    --dropout 0.2 \
    --lr 1e-4 \
    --wandb \
    --wandb_project engiopt-ddm-3d
