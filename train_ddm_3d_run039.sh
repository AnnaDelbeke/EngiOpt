#!/bin/bash
#SBATCH --job-name=train_ddm_3d_run039
#SBATCH --output=logs/train_ddm_3d_run039_%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.train_ddm_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --bae_latent_dim 128 \
    --model_name     ddm_3d_run039_v1 \
    --w_aoa          9.0
