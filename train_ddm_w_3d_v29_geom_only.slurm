#!/bin/bash
#SBATCH --job-name=ddm_w_3d_v29_geom_only
#SBATCH --output=logs/ddm_w_3d_v29_geom_only_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --nice=0

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d_geom_only/lvae_3d_v29_geom_only_best.pth \
    --model_name ddm_w_3d_v29_geom_only \
    --lae_latent_dim 64 \
    --w_aoa 9.0 \
    --lr 1e-4 \
    --save_dir results/ddm_w_3d_geom_only \
    --wandb \
    --wandb_project engiopt-ddm-w-3d
