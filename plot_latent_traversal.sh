#!/bin/bash
#SBATCH --job-name=plot_latent_traversal
#SBATCH --output=logs/plot_latent_traversal_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.plot_latent_traversal
