#!/bin/bash
#SBATCH --job-name=eval_lvae_3d_v2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lvae_3d_v2_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.lvae.evaluate_lvae_3d \
    --checkpoint     results/lvae_3d/lvae_3d_v4_best.pth \
    --bae_checkpoint results/bezier_ae_3d/run_014/models/bezier_ae_3d_best.pt \
    --bae_latent_dim 256 \
    --lae_latent_dim 64 \
    --n_spans        15 \
    --dropout        0.2 \
    --cpx_bound      -0.1 1.0 \
    --cpy_bound      -0.3 0.3 \
    --save_dir       results/lvae_3d_evaluation/v2
