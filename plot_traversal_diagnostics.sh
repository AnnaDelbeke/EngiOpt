#!/bin/bash
#SBATCH --job-name=plot_traversal_diag
#SBATCH --output=logs/plot_traversal_diagnostics_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_traversal_diagnostics.py
