#!/bin/bash
#SBATCH --job-name=plot_latent_dims
#SBATCH --output=logs/plot_latent_dims_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_latent_dims.py
