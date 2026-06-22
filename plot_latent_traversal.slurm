#!/bin/bash
#SBATCH --job-name=plot_traversal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/plot_traversal_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.plot_latent_traversal \
    --checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --save_stem thesis/figures/latent_traversal \
    --n_steps 5
