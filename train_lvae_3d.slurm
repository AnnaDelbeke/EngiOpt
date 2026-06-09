#!/bin/bash
#SBATCH --job-name=train_lvae_3d
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/train_lvae_3d_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint results/bezier_ae_3d/run_014/models/bezier_ae_3d_best.pt \
    --bae_latent_dim 256 \
    --model_name lvae_3d_v5 \
    --n_epochs 10000 \
    --lr 1e-3 \
    --lae_latent_dim 128 \
    --lambda_lv 1e-3 \
    --dropout 0.2 \
    --prune_every 500 \
    --prune_threshold 0.02 \
    --wandb
