#!/bin/bash
#SBATCH --job-name=eval_bae_3d_run20
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_bae_3d_run20_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
    --checkpoint results/bezier_ae_3d/run_020/models/bezier_ae_3d_best.pt \
    --run_dir    results/bezier_ae_3d/run_020 \
    --n_examples 5 \
    --latent_dim 256 \
    --cpx_bound -0.1 1.0 \
    --cpy_bound -0.22 0.30
