#!/bin/bash
#SBATCH --job-name=latent_std_plot
#SBATCH --output=logs/latent_std_replot_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python plot_latent_std_thesis.py
