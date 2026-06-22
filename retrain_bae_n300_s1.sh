#!/bin/bash
#SBATCH --job-name=bae3d_retrain_n300_s1
#SBATCH --output=logs/bae3d_retrain_n300_s1_%j.out
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -m engiopt.bezier_ae.train_bezier_ae_3d_ablation \
    --n_samples 300 \
    --seed 1
