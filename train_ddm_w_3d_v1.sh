#!/bin/bash
#SBATCH --job-name=train_ddm_w_3d_v1
#SBATCH --output=logs/train_ddm_w_3d_v1_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v8_best.pth \
    --model_name ddm_w_3d_v1 \
    --bae_latent_dim 64 \
    --lae_latent_dim 64 \
    --w_aoa 9.0 \
    --lr 1e-4 \
    --wandb \
    --wandb_project engiopt-ddm-w-3d
