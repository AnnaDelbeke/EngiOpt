#!/bin/bash
#SBATCH --job-name=latent_corr
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/latent_corr_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u analyze_latent_correlations.py \
    --out thesis/figures/latent_correlations.pdf
