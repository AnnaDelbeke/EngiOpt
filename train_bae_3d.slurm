#!/bin/bash
#SBATCH --job-name=train_bae_3d
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/train_bae_3d_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.bezier_ae.train_bezier_ae_3d
# After training, run evaluation:
# python -u -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
#     --checkpoint results/bezier_ae_3d/run_014/models/bezier_ae_3d_best.pt \
#     --run_dir    results/bezier_ae_3d/run_014
